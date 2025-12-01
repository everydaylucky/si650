#!/usr/bin/env python3
"""
训练LightGBM Learning-to-Rank模型
"""
import sys
import argparse
import json
import traceback
from pathlib import Path
from tqdm import tqdm
import numpy as np
import lightgbm as lgb
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features import FeatureExtractor
from src.features.ir_features import IRFeatureExtractor
from src.models.retrieval.bm25_retriever import BM25Retriever
from src.models.retrieval.tfidf_retriever import TFIDFRetriever
from src.models.retrieval.dense_retriever import DenseRetriever

def extract_features_for_training(train_file: str, 
                                  feature_extractor: FeatureExtractor,
                                  variant: str = "zero-shot",
                                  use_fine_tuned: bool = False):
    """为训练提取特征"""
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 收集所有文档用于构建索引
    all_documents = []
    for sample in data:
        if "target_paper" in sample:
            all_documents.append(sample["target_paper"])
        negatives = sample.get("negatives", []) or []
        for neg in negatives:
            if isinstance(neg, dict):
                all_documents.append(neg)
    
    # 去重（基于id）
    seen_ids = set()
    unique_docs = []
    for doc in all_documents:
        doc_id = doc.get("id") or doc.get("paper_id")
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_docs.append(doc)
    
    print(f"构建索引 ({len(unique_docs)} 个文档)...", flush=True)
    
    # 初始化检索器并构建索引
    print("初始化BM25检索器...", flush=True)
    bm25 = BM25Retriever()
    print("构建BM25索引...", flush=True)
    bm25.build_index(unique_docs)
    print("✓ BM25索引构建完成", flush=True)
    
    print("初始化TF-IDF检索器...", flush=True)
    tfidf = TFIDFRetriever()
    print("构建TF-IDF索引...", flush=True)
    tfidf.build_index(unique_docs)
    print("✓ TF-IDF索引构建完成", flush=True)
    
    # 对于zero-shot L2R，不需要DenseRetriever（嵌入特征都是占位符）
    # 对于fine-tuned，使用fine-tuned模型的embedding
    dense = None
    scibert_ft = None
    if use_fine_tuned:
        print("初始化Fine-tuned模型用于embedding特征...", flush=True)
        
        # 使用Fine-tuned SPECTER2
        specter2_ft_path = project_root / "experiments" / "checkpoints" / "specter2"
        if specter2_ft_path.exists():
            print(f"  加载SPECTER2 Fine-tuned: {specter2_ft_path}", flush=True)
            dense = DenseRetriever(
                model_name="allenai/specter2_base",
                fine_tuned_path=str(specter2_ft_path)
            )
            dense.build_index(unique_docs)
            print("✓ SPECTER2 Fine-tuned索引构建完成", flush=True)
        else:
            print(f"⚠ SPECTER2 Fine-tuned模型不存在: {specter2_ft_path}，使用zero-shot", flush=True)
            dense = DenseRetriever()
            dense.build_index(unique_docs)
        
        # 使用Fine-tuned SciBERT（用于计算相似度）
        scibert_ft_path = project_root / "experiments" / "checkpoints" / "scibert"
        if scibert_ft_path.exists():
            print(f"  加载SciBERT Fine-tuned: {scibert_ft_path}", flush=True)
            from src.models.reranking.bi_encoder import BiEncoder
            scibert_ft = BiEncoder(
                model_name="allenai/scibert_scivocab_uncased",
                fine_tuned_path=str(scibert_ft_path)
            )
            print("✓ SciBERT Fine-tuned加载完成", flush=True)
        else:
            print(f"⚠ SciBERT Fine-tuned模型不存在: {scibert_ft_path}，跳过", flush=True)
    else:
        print("跳过Dense检索器（zero-shot模式下嵌入特征为占位符）", flush=True)
    
    # 更新IR特征提取器
    feature_extractor.ir_features.bm25 = bm25
    feature_extractor.ir_features.tfidf = tfidf
    
    X = []
    y = []
    qids = []
    
    print(f"开始提取特征，共 {len(data)} 个样本...", flush=True)
    query_id = 0
    for sample in tqdm(data, desc="提取特征", mininterval=1.0):
        citation_context = sample.get("citation_context", {}).get("text", "") if isinstance(sample.get("citation_context"), dict) else sample.get("citation_context", "")
        
        query = {
            "citation_context": citation_context,
            "source_paper_id": sample.get("source_paper_id", ""),
            "source_categories": sample.get("source_categories", []) or sample.get("source_paper", {}).get("categories", []),
            "source_year": sample.get("source_year") or sample.get("source_paper", {}).get("year", 2024)
        }
        
        # 获取候选文档的检索分数
        candidates = []
        
        # 正样本
        positive = sample.get("target_paper", {})
        if positive:
            candidates.append((positive, 1))
        
        # 负样本
        negatives = sample.get("negatives", []) or []
        for neg in negatives[:20]:  # 限制负样本数量以提高训练速度
            if isinstance(neg, dict):
                candidates.append((neg, 0))
        
        # 批量获取IR分数（避免重复检索）
        # 只对每个query检索一次，然后查找候选文档的分数
        if citation_context and candidates:
            # 获取所有候选文档的ID
            candidate_ids = set()
            for candidate, _ in candidates:
                cid = candidate.get("id") or candidate.get("paper_id")
                if cid:
                    candidate_ids.add(cid)
            
            # 检索top-K结果（K要足够大以包含所有候选）
            top_k = max(1000, len(candidate_ids) * 2)
            bm25_results = bm25.retrieve(citation_context, top_k=top_k)
            tfidf_results = tfidf.retrieve(citation_context, top_k=top_k)
            
            # 转换为字典以便快速查找
            bm25_scores_dict = {pid: score for pid, score in bm25_results}
            tfidf_scores_dict = {pid: score for pid, score in tfidf_results}
        else:
            bm25_scores_dict = {}
            tfidf_scores_dict = {}
        
        # 为每个候选文档提取特征
        for candidate, label in candidates:
            candidate_id = candidate.get("id") or candidate.get("paper_id")
            
            # 获取IR分数（如果候选不在top-K中，分数为0）
            bm25_score = bm25_scores_dict.get(candidate_id, 0.0)
            tfidf_score = tfidf_scores_dict.get(candidate_id, 0.0)
            
            # 更新query中的分数（用于嵌入特征）
            query["_bm25_score"] = bm25_score
            query["_tfidf_score"] = tfidf_score
            
            # 如果使用fine-tuned模型，计算embedding相似度
            if use_fine_tuned:
                # SPECTER2相似度
                if dense:
                    candidate_text = f"{candidate.get('title', '')} {candidate.get('abstract', '')}".strip()
                    query_text = query.get("citation_context", "")
                    try:
                        # 使用DenseRetriever计算相似度
                        results = dense.retrieve(query_text, top_k=1000)
                        specter2_score = next((s for pid, s in results if pid == candidate_id), 0.0)
                        query["_specter2_score"] = specter2_score
                    except:
                        query["_specter2_score"] = 0.0
                
                # SciBERT相似度
                if scibert_ft:
                    candidate_text = f"{candidate.get('title', '')} {candidate.get('abstract', '')}".strip()
                    query_text = query.get("citation_context", "")
                    try:
                        # 编码并计算余弦相似度
                        query_emb = scibert_ft.model.encode([query_text], convert_to_numpy=True)[0]
                        cand_emb = scibert_ft.model.encode([candidate_text], convert_to_numpy=True)[0]
                        import numpy as np
                        scibert_score = np.dot(query_emb, cand_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(cand_emb))
                        query["_scibert_score"] = float(scibert_score)
                    except:
                        query["_scibert_score"] = 0.0
            
            # 提取所有特征
            features = feature_extractor.extract_all_features(query, candidate)
            X.append(features)
            y.append(label)
            qids.append(query_id)
        
        query_id += 1
    
    return np.array(X), np.array(y), np.array(qids)

def main():
    parser = argparse.ArgumentParser(description='训练LightGBM L2R模型')
    parser.add_argument('--train_file', type=str, default='data/processed/fast_experiment/train.json')
    parser.add_argument('--val_file', type=str, default='data/processed/fast_experiment/val.json')
    parser.add_argument('--output_dir', type=str, default='experiments/checkpoints/l2r')
    parser.add_argument('--variant', type=str, default='zero-shot', choices=['zero-shot', 'fine-tuned'],
                        help='模型变体: zero-shot 或 fine-tuned')
    parser.add_argument('--num_leaves', type=int, default=31)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--n_estimators', type=int, default=300)
    parser.add_argument('--max_depth', type=int, default=6)
    
    args = parser.parse_args()
    
    train_file = project_root / args.train_file
    if not train_file.exists():
        print(f"❌ 训练文件不存在: {train_file}")
        return
    
    # 根据variant设置输出目录
    variant_suffix = "ft" if args.variant == "fine-tuned" else "zs"
    output_dir = project_root / args.output_dir / variant_suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"训练LightGBM L2R模型 ({args.variant})")
    print("=" * 60)
    
    try:
        # 初始化特征提取器
        print("\n初始化特征提取器...", flush=True)
        feature_extractor = FeatureExtractor()
        print("✓ 特征提取器初始化成功", flush=True)
        
        # 提取特征
        print("\n提取训练特征...", flush=True)
        use_fine_tuned = (args.variant == "fine-tuned")
        X_train, y_train, qids_train = extract_features_for_training(
            str(train_file), feature_extractor, args.variant, use_fine_tuned
        )
        print(f"✓ 训练特征: {X_train.shape}", flush=True)
        print(f"  正样本: {np.sum(y_train == 1)}, 负样本: {np.sum(y_train == 0)}", flush=True)
        
        # 计算每个query的文档数量（用于group参数）
        query_groups = defaultdict(int)
        for qid in qids_train:
            query_groups[qid] += 1
        groups = [query_groups[qid] for qid in sorted(query_groups.keys())]
        
        # 按query_id排序数据（LightGBM要求）
        sorted_indices = np.argsort(qids_train)
        X_train_sorted = X_train[sorted_indices]
        y_train_sorted = y_train[sorted_indices]
        qids_train_sorted = qids_train[sorted_indices]
        
        # 重新计算groups（基于排序后的数据）
        groups = []
        current_qid = qids_train_sorted[0]
        current_count = 0
        for qid in qids_train_sorted:
            if qid == current_qid:
                current_count += 1
            else:
                groups.append(current_count)
                current_count = 1
                current_qid = qid
        groups.append(current_count)  # 最后一个query
        
        print(f"  查询数量: {len(groups)}", flush=True)
        print(f"  平均每个查询的文档数: {np.mean(groups):.1f}", flush=True)
        
        # 创建L2R数据集
        train_data = lgb.Dataset(
            X_train_sorted,
            label=y_train_sorted,
            group=groups
        )
        
        # 如果有验证集，也处理
        val_data = None
        if args.val_file:
            val_file = project_root / args.val_file
            if val_file.exists():
                print("\n提取验证特征...", flush=True)
                X_val, y_val, qids_val = extract_features_for_training(
                    str(val_file), feature_extractor, args.variant, use_fine_tuned
                )
                
                # 排序验证数据
                sorted_indices_val = np.argsort(qids_val)
                X_val_sorted = X_val[sorted_indices_val]
                y_val_sorted = y_val[sorted_indices_val]
                qids_val_sorted = qids_val[sorted_indices_val]
                
                # 计算验证集groups
                groups_val = []
                if len(qids_val_sorted) > 0:
                    current_qid = qids_val_sorted[0]
                    current_count = 0
                    for qid in qids_val_sorted:
                        if qid == current_qid:
                            current_count += 1
                        else:
                            groups_val.append(current_count)
                            current_count = 1
                            current_qid = qid
                    groups_val.append(current_count)
                
                val_data = lgb.Dataset(
                    X_val_sorted,
                    label=y_val_sorted,
                    group=groups_val,
                    reference=train_data
                )
                print(f"✓ 验证特征: {X_val_sorted.shape}")
        
        # 训练参数
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'num_leaves': args.num_leaves,
            'learning_rate': args.learning_rate,
            'max_depth': args.max_depth,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': 1
        }
        
        # 训练
        print("\n开始训练LightGBM模型...", flush=True)
        valid_sets = [train_data]
        if val_data:
            valid_sets.append(val_data)
            valid_names = ['train', 'val']
        else:
            valid_names = ['train']
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=args.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # 保存模型
        model_path = output_dir / "l2r_model.txt"
        model.save_model(str(model_path))
        print(f"\n✓ 模型已保存到: {model_path}", flush=True)
        
        return str(model_path)
        
    except Exception as e:
        print(f"\n❌ 训练失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

