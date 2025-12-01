#!/usr/bin/env python3
"""
快速系统测试脚本
用于排查潜在问题
"""
import sys
import json
import traceback
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

def test_imports():
    """测试所有关键模块导入"""
    print("=" * 80)
    print("测试1: 模块导入")
    print("=" * 80)
    errors = []
    
    try:
        from src.models.retrieval import BM25Retriever, TFIDFRetriever, DenseRetriever
        print("✓ 检索模型导入成功")
    except Exception as e:
        errors.append(f"检索模型导入失败: {e}")
        print(f"✗ 检索模型导入失败: {e}")
    
    try:
        from src.models.reranking import BiEncoder, ReciprocalRankFusion
        print("✓ 重排序模型导入成功")
    except Exception as e:
        errors.append(f"重排序模型导入失败: {e}")
        print(f"✗ 重排序模型导入失败: {e}")
    
    try:
        from src.models.ranking import CrossEncoderRanker, L2RRanker
        print("✓ 排序模型导入成功")
    except Exception as e:
        errors.append(f"排序模型导入失败: {e}")
        print(f"✗ 排序模型导入失败: {e}")
    
    try:
        from src.pipeline.multi_stage_pipeline import MultiStagePipeline
        print("✓ Pipeline导入成功")
    except Exception as e:
        errors.append(f"Pipeline导入失败: {e}")
        print(f"✗ Pipeline导入失败: {e}")
    
    try:
        from src.features import FeatureExtractor
        print("✓ 特征提取器导入成功")
    except Exception as e:
        errors.append(f"特征提取器导入失败: {e}")
        print(f"✗ 特征提取器导入失败: {e}")
    
    try:
        from src.evaluation import Evaluator
        print("✓ 评估器导入成功")
    except Exception as e:
        errors.append(f"评估器导入失败: {e}")
        print(f"✗ 评估器导入失败: {e}")
    
    return len(errors) == 0, errors

def test_data_files():
    """测试数据文件"""
    print("\n" + "=" * 80)
    print("测试2: 数据文件")
    print("=" * 80)
    errors = []
    
    data_dir = project_root / "data" / "processed" / "fast_experiment"
    
    for split in ["train", "val", "test"]:
        file_path = data_dir / f"{split}.json"
        if not file_path.exists():
            errors.append(f"数据文件不存在: {file_path}")
            print(f"✗ {split}.json 不存在")
            continue
        
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                errors.append(f"{split}.json 格式错误: 不是列表")
                print(f"✗ {split}.json 格式错误: 不是列表")
                continue
            
            if len(data) == 0:
                errors.append(f"{split}.json 为空")
                print(f"✗ {split}.json 为空")
                continue
            
            # 检查必要字段
            sample = data[0]
            required_fields = ["citation_context", "target_paper"]
            missing = [f for f in required_fields if f not in sample]
            if missing:
                errors.append(f"{split}.json 缺少字段: {missing}")
                print(f"✗ {split}.json 缺少字段: {missing}")
            else:
                print(f"✓ {split}.json: {len(data)} 个样本")
                
        except Exception as e:
            errors.append(f"{split}.json 读取失败: {e}")
            print(f"✗ {split}.json 读取失败: {e}")
    
    return len(errors) == 0, errors

def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 80)
    print("测试3: 模型加载")
    print("=" * 80)
    errors = []
    
    # 测试BM25
    try:
        from src.models.retrieval import BM25Retriever
        bm25 = BM25Retriever()
        print("✓ BM25初始化成功")
    except Exception as e:
        errors.append(f"BM25初始化失败: {e}")
        print(f"✗ BM25初始化失败: {e}")
    
    # 测试TF-IDF
    try:
        from src.models.retrieval import TFIDFRetriever
        tfidf = TFIDFRetriever()
        print("✓ TF-IDF初始化成功")
    except Exception as e:
        errors.append(f"TF-IDF初始化失败: {e}")
        print(f"✗ TF-IDF初始化失败: {e}")
    
    # 测试DenseRetriever (SPECTER2) - 只测试初始化，不加载模型
    try:
        from src.models.retrieval import DenseRetriever
        # 不实际加载模型，只测试类定义
        print("✓ DenseRetriever类定义正常")
    except Exception as e:
        errors.append(f"DenseRetriever类定义失败: {e}")
        print(f"✗ DenseRetriever类定义失败: {e}")
    
    # 测试BiEncoder (SciBERT) - 只测试初始化
    try:
        from src.models.reranking import BiEncoder
        print("✓ BiEncoder类定义正常")
    except Exception as e:
        errors.append(f"BiEncoder类定义失败: {e}")
        print(f"✗ BiEncoder类定义失败: {e}")
    
    # 测试CrossEncoderRanker
    try:
        from src.models.ranking import CrossEncoderRanker
        print("✓ CrossEncoderRanker类定义正常")
    except Exception as e:
        errors.append(f"CrossEncoderRanker类定义失败: {e}")
        print(f"✗ CrossEncoderRanker类定义失败: {e}")
    
    # 测试L2RRanker
    try:
        from src.models.ranking import L2RRanker
        print("✓ L2RRanker类定义正常")
    except Exception as e:
        errors.append(f"L2RRanker类定义失败: {e}")
        print(f"✗ L2RRanker类定义失败: {e}")
    
    return len(errors) == 0, errors

def test_feature_extraction():
    """测试特征提取"""
    print("\n" + "=" * 80)
    print("测试4: 特征提取")
    print("=" * 80)
    errors = []
    
    try:
        from src.features import FeatureExtractor
        from src.models.retrieval import BM25Retriever, TFIDFRetriever
        
        extractor = FeatureExtractor()
        
        # 创建测试数据
        test_docs = [
            {"id": "doc1", "title": "Machine Learning", "abstract": "This paper discusses machine learning algorithms.", "categories": ["cs.LG"], "year": 2023},
            {"id": "doc2", "title": "Deep Learning", "abstract": "Deep neural networks for computer vision.", "categories": ["cs.CV"], "year": 2024}
        ]
        
        # 初始化检索器
        bm25 = BM25Retriever()
        bm25.build_index(test_docs)
        extractor.ir_features.bm25 = bm25
        
        tfidf = TFIDFRetriever()
        tfidf.build_index(test_docs)
        extractor.ir_features.tfidf = tfidf
        
        # 测试特征提取
        query = {
            "citation_context": "Recent work on machine learning",
            "source_categories": ["cs.LG"],
            "source_year": 2024
        }
        
        candidate = test_docs[0]
        features = extractor.extract_all_features(query, candidate)
        
        if len(features) == 0:
            errors.append("特征提取返回空数组")
            print(f"✗ 特征提取返回空数组")
        elif len(features) != 18:
            errors.append(f"特征数量不正确: 期望18，实际{len(features)}")
            print(f"✗ 特征数量不正确: 期望18，实际{len(features)}")
        else:
            print(f"✓ 特征提取成功: {len(features)} 维特征")
            
    except Exception as e:
        errors.append(f"特征提取测试失败: {e}")
        print(f"✗ 特征提取测试失败: {e}")
        traceback.print_exc()
    
    return len(errors) == 0, errors

def test_pipeline_basic():
    """测试基础Pipeline"""
    print("\n" + "=" * 80)
    print("测试5: Pipeline基础功能")
    print("=" * 80)
    errors = []
    
    try:
        from src.pipeline.multi_stage_pipeline import MultiStagePipeline
        
        # 创建测试配置
        config = {
            "stage1": {
                "use_bm25": True,
                "use_specter2": True,
                "top_k": 1000
            },
            "stage2": {
                "use_rrf": True,
                "top_k": 50
            },
            "stage3": {
                "use_cross_encoder": True,
                "top_k": 20
            }
        }
        
        pipeline = MultiStagePipeline(config)
        print("✓ Pipeline初始化成功")
        
        # 测试文档索引构建（需要至少100个文档用于FAISS聚类）
        # 如果文档少，只测试BM25，不测试SPECTER2
        test_docs = [
            {"id": f"doc{i}", "title": f"Paper {i}", "abstract": f"This is paper {i} about machine learning.", "categories": ["cs.LG"], "year": 2023}
            for i in range(1, 101)  # 生成100个文档
        ]
        
        pipeline.build_indices(test_docs)
        print("✓ Pipeline索引构建成功")
        
        # 测试检索
        query = {
            "citation_context": "Recent work on machine learning",
            "source_categories": ["cs.LG"],
            "source_year": 2024
        }
        
        results = pipeline.retrieve(query)
        
        if not results:
            errors.append("Pipeline检索返回空结果")
            print(f"✗ Pipeline检索返回空结果")
        elif len(results) == 0:
            errors.append("Pipeline检索返回空列表")
            print(f"✗ Pipeline检索返回空列表")
        else:
            print(f"✓ Pipeline检索成功: 返回 {len(results)} 个结果")
            print(f"  前3个结果: {results[:3]}")
            
    except Exception as e:
        errors.append(f"Pipeline测试失败: {e}")
        print(f"✗ Pipeline测试失败: {e}")
        traceback.print_exc()
    
    return len(errors) == 0, errors

def test_evaluation():
    """测试评估指标计算"""
    print("\n" + "=" * 80)
    print("测试6: 评估指标")
    print("=" * 80)
    errors = []
    
    try:
        from src.evaluation import Evaluator
        
        evaluator = Evaluator()
        
        # 创建测试数据
        ground_truth = ["doc1", "doc2", "doc3"]
        predictions = [
            [("doc1", 0.9), ("doc4", 0.8), ("doc2", 0.7)],  # doc1在位置1
            [("doc5", 0.9), ("doc2", 0.8), ("doc1", 0.7)],  # doc2在位置2
            [("doc3", 0.9), ("doc1", 0.8), ("doc2", 0.7)]   # doc3在位置1
        ]
        
        metrics = evaluator.evaluate(predictions, ground_truth)
        
        if "mrr" not in metrics:
            errors.append("MRR指标缺失")
            print(f"✗ MRR指标缺失")
        elif metrics["mrr"] == 0:
            errors.append("MRR计算结果为0（可能有问题）")
            print(f"✗ MRR计算结果为0")
        else:
            print(f"✓ MRR计算成功: {metrics['mrr']:.4f}")
        
        if "recall@10" not in metrics:
            errors.append("Recall@10指标缺失")
            print(f"✗ Recall@10指标缺失")
        else:
            print(f"✓ Recall@10计算成功: {metrics['recall@10']:.4f}")
        
        if "ndcg@10" not in metrics:
            errors.append("NDCG@10指标缺失")
            print(f"✗ NDCG@10指标缺失")
        else:
            print(f"✓ NDCG@10计算成功: {metrics['ndcg@10']:.4f}")
            
    except Exception as e:
        errors.append(f"评估指标测试失败: {e}")
        print(f"✗ 评估指标测试失败: {e}")
        traceback.print_exc()
    
    return len(errors) == 0, errors

def test_config_files():
    """测试配置文件"""
    print("\n" + "=" * 80)
    print("测试7: 配置文件")
    print("=" * 80)
    errors = []
    
    import yaml
    from src.experiments.experiment_config import ALL_EXPERIMENTS
    
    for exp_id, exp_config in ALL_EXPERIMENTS.items():
        config_path = project_root / exp_config.config_path
        
        if not config_path.exists():
            errors.append(f"{exp_id}: 配置文件不存在 {config_path}")
            print(f"✗ {exp_id}: 配置文件不存在")
            continue
        
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            if not config:
                errors.append(f"{exp_id}: 配置文件为空")
                print(f"✗ {exp_id}: 配置文件为空")
                continue
            
            # 检查必要字段
            if "stage1" not in config:
                errors.append(f"{exp_id}: 缺少stage1配置")
                print(f"✗ {exp_id}: 缺少stage1配置")
            else:
                print(f"✓ {exp_id}: 配置文件正常")
                
        except Exception as e:
            errors.append(f"{exp_id}: 配置文件读取失败: {e}")
            print(f"✗ {exp_id}: 配置文件读取失败: {e}")
    
    return len(errors) == 0, errors

def test_trained_models():
    """测试已训练的模型"""
    print("\n" + "=" * 80)
    print("测试8: 已训练模型")
    print("=" * 80)
    errors = []
    
    checkpoints_dir = project_root / "experiments" / "checkpoints"
    
    # 检查SciBERT
    scibert_path = checkpoints_dir / "scibert"
    if (scibert_path / "config.json").exists():
        print(f"✓ SciBERT模型存在: {scibert_path}")
    else:
        print(f"ℹ SciBERT模型不存在（需要训练）")
    
    # 检查Cross-Encoder
    crossenc_path = checkpoints_dir / "cross_encoder"
    if (crossenc_path / "config.json").exists() or (crossenc_path / "pytorch_model.bin").exists():
        print(f"✓ Cross-Encoder模型存在: {crossenc_path}")
    else:
        print(f"ℹ Cross-Encoder模型不存在（需要训练）")
    
    # 检查L2R
    l2r_zs_path = checkpoints_dir / "l2r" / "zs" / "l2r_model.txt"
    l2r_ft_path = checkpoints_dir / "l2r" / "ft" / "l2r_model.txt"
    
    if l2r_zs_path.exists():
        print(f"✓ L2R Zero-shot模型存在: {l2r_zs_path}")
    else:
        print(f"ℹ L2R Zero-shot模型不存在")
    
    if l2r_ft_path.exists():
        print(f"✓ L2R Fine-tuned模型存在: {l2r_ft_path}")
    else:
        print(f"ℹ L2R Fine-tuned模型不存在（需要训练）")
    
    return True, []

def test_pipeline_configs():
    """测试Pipeline配置是否正确"""
    print("\n" + "=" * 80)
    print("测试9: Pipeline配置检查")
    print("=" * 80)
    errors = []
    
    import yaml
    
    # 检查pipeline basic
    basic_config_path = project_root / "config" / "experiments" / "exp_5_1_pipeline_basic.yaml"
    if basic_config_path.exists():
        with open(basic_config_path) as f:
            config = yaml.safe_load(f)
        
        stage1 = config.get("stage1", {})
        stage2 = config.get("stage2", {})
        stage3 = config.get("stage3", {})
        
        # 检查是否有启用的检索器
        has_stage1 = any([stage1.get("use_bm25", False), 
                         stage1.get("use_specter2", False),
                         stage1.get("use_tfidf", False)])
        has_stage2 = any([stage2.get("use_rrf", False),
                         stage2.get("use_bi_encoder", False),
                         stage2.get("use_colbert", False)])
        has_stage3 = any([stage3.get("use_cross_encoder", False),
                         stage3.get("use_l2r", False)])
        
        if not has_stage1 and not has_stage2 and not has_stage3:
            print("⚠ Pipeline Basic配置中所有阶段都未启用（会在运行时自动修复）")
        else:
            print(f"✓ Pipeline Basic配置: Stage1={has_stage1}, Stage2={has_stage2}, Stage3={has_stage3}")
    
    # 检查pipeline optimized
    opt_config_path = project_root / "config" / "experiments" / "exp_5_2_pipeline_optimized.yaml"
    if opt_config_path.exists():
        with open(opt_config_path) as f:
            config = yaml.safe_load(f)
        
        stage1 = config.get("stage1", {})
        stage2 = config.get("stage2", {})
        stage3 = config.get("stage3", {})
        
        has_stage1 = any([stage1.get("use_bm25", False), 
                         stage1.get("use_specter2", False),
                         stage1.get("use_tfidf", False)])
        has_stage2 = any([stage2.get("use_rrf", False),
                         stage2.get("use_bi_encoder", False),
                         stage2.get("use_colbert", False)])
        has_stage3 = any([stage3.get("use_cross_encoder", False),
                         stage3.get("use_l2r", False)])
        
        if not has_stage1 and not has_stage2 and not has_stage3:
            print("⚠ Pipeline Optimized配置中所有阶段都未启用（会在运行时自动修复）")
        else:
            print(f"✓ Pipeline Optimized配置: Stage1={has_stage1}, Stage2={has_stage2}, Stage3={has_stage3}")
    
    return True, []

def test_l2r_model_loading():
    """测试L2R模型加载"""
    print("\n" + "=" * 80)
    print("测试10: L2R模型加载")
    print("=" * 80)
    errors = []
    
    from src.models.ranking import L2RRanker
    
    # 检查zero-shot模型
    l2r_zs_path = project_root / "experiments" / "checkpoints" / "l2r" / "zs" / "l2r_model.txt"
    if l2r_zs_path.exists():
        try:
            l2r = L2RRanker(model_path=str(l2r_zs_path))
            print(f"✓ L2R Zero-shot模型加载成功: {l2r_zs_path}")
        except Exception as e:
            errors.append(f"L2R Zero-shot模型加载失败: {e}")
            print(f"✗ L2R Zero-shot模型加载失败: {e}")
            traceback.print_exc()
    else:
        print(f"ℹ L2R Zero-shot模型不存在: {l2r_zs_path}")
    
    # 检查fine-tuned模型
    l2r_ft_path = project_root / "experiments" / "checkpoints" / "l2r" / "ft" / "l2r_model.txt"
    if l2r_ft_path.exists():
        try:
            l2r = L2RRanker(model_path=str(l2r_ft_path))
            print(f"✓ L2R Fine-tuned模型加载成功: {l2r_ft_path}")
        except Exception as e:
            errors.append(f"L2R Fine-tuned模型加载失败: {e}")
            print(f"✗ L2R Fine-tuned模型加载失败: {e}")
            traceback.print_exc()
    else:
        print(f"ℹ L2R Fine-tuned模型不存在: {l2r_ft_path}")
    
    return len(errors) == 0, errors

def test_data_format():
    """测试数据格式"""
    print("\n" + "=" * 80)
    print("测试11: 数据格式检查")
    print("=" * 80)
    errors = []
    
    data_dir = project_root / "data" / "processed" / "fast_experiment"
    test_file = data_dir / "test.json"
    
    if not test_file.exists():
        errors.append("测试文件不存在")
        print(f"✗ 测试文件不存在: {test_file}")
        return False, errors
    
    try:
        with open(test_file) as f:
            data = json.load(f)
        
        if len(data) == 0:
            errors.append("测试数据为空")
            print(f"✗ 测试数据为空")
            return False, errors
        
        # 检查前几个样本的格式
        sample = data[0]
        
        # 检查citation_context格式
        citation_context = sample.get("citation_context", "")
        if isinstance(citation_context, dict):
            if "text" not in citation_context:
                errors.append("citation_context是字典但缺少text字段")
                print(f"✗ citation_context格式问题: 是字典但缺少text字段")
            else:
                print(f"✓ citation_context格式: 字典格式（包含text字段）")
        elif isinstance(citation_context, str):
            print(f"✓ citation_context格式: 字符串格式")
        else:
            errors.append(f"citation_context格式未知: {type(citation_context)}")
            print(f"✗ citation_context格式未知: {type(citation_context)}")
        
        # 检查target_paper格式
        if "target_paper" in sample:
            target = sample["target_paper"]
            if not isinstance(target, dict):
                errors.append("target_paper不是字典")
                print(f"✗ target_paper不是字典")
            elif "id" not in target:
                errors.append("target_paper缺少id字段")
                print(f"✗ target_paper缺少id字段")
            else:
                print(f"✓ target_paper格式正常")
        
        # 检查target_paper_id
        if "target_paper_id" in sample:
            print(f"✓ target_paper_id字段存在")
        
    except Exception as e:
        errors.append(f"数据格式检查失败: {e}")
        print(f"✗ 数据格式检查失败: {e}")
        traceback.print_exc()
    
    return len(errors) == 0, errors

def test_pipeline_retrieval():
    """测试Pipeline检索功能"""
    print("\n" + "=" * 80)
    print("测试12: Pipeline检索功能")
    print("=" * 80)
    errors = []
    
    try:
        from src.pipeline.multi_stage_pipeline import MultiStagePipeline
        
        # 加载测试数据
        data_dir = project_root / "data" / "processed" / "fast_experiment"
        test_file = data_dir / "test.json"
        
        if not test_file.exists():
            print("ℹ 跳过Pipeline检索测试（测试文件不存在）")
            return True, []
        
        with open(test_file) as f:
            test_data = json.load(f)
        
        if len(test_data) == 0:
            print("ℹ 跳过Pipeline检索测试（测试数据为空）")
            return True, []
        
        # 创建配置（使用SPECTER2）
        config = {
            "stage1": {
                "use_bm25": True,
                "use_specter2": True,
                "top_k": 100
            },
            "stage2": {
                "top_k": 50
            },
            "stage3": {
                "top_k": 20
            }
        }
        
        pipeline = MultiStagePipeline(config)
        
        # 构建索引（使用所有文档，但至少需要100个用于FAISS）
        documents = []
        for item in test_data:
            if "target_paper" in item:
                documents.append(item["target_paper"])
        
        if len(documents) == 0:
            print("ℹ 跳过Pipeline检索测试（没有可用文档）")
            return True, []
        
        # 如果文档少于100个，只测试BM25，不测试SPECTER2
        if len(documents) < 100:
            print(f"⚠ 文档数({len(documents)})少于100，将只测试BM25，跳过SPECTER2")
            config["stage1"]["use_specter2"] = False
        
        pipeline.build_indices(documents)
        print(f"✓ Pipeline索引构建成功: {len(documents)} 个文档")
        
        # 测试检索
        sample = test_data[0]
        query = {
            "citation_context": sample.get("citation_context", ""),
            "source_paper_id": sample.get("source_paper_id", ""),
            "source_categories": sample.get("source_categories", []),
            "source_year": sample.get("source_year", 2024)
        }
        
        # 处理citation_context可能是字典的情况
        if isinstance(query["citation_context"], dict):
            query["citation_context"] = query["citation_context"].get("text", "")
        
        results = pipeline.retrieve(query)
        
        if not results:
            errors.append("Pipeline检索返回空结果")
            print(f"✗ Pipeline检索返回空结果")
        elif len(results) == 0:
            errors.append("Pipeline检索返回空列表")
            print(f"✗ Pipeline检索返回空列表")
        else:
            print(f"✓ Pipeline检索成功: 返回 {len(results)} 个结果")
            print(f"  前3个结果ID: {[r[0] for r in results[:3]]}")
            
    except Exception as e:
        errors.append(f"Pipeline检索测试失败: {e}")
        print(f"✗ Pipeline检索测试失败: {e}")
        traceback.print_exc()
    
    return len(errors) == 0, errors

def main():
    """运行所有测试"""
    print("=" * 80)
    print("系统快速测试")
    print("=" * 80)
    print("")
    
    all_errors = []
    test_results = []
    
    # 运行所有测试
    tests = [
        ("模块导入", test_imports),
        ("数据文件", test_data_files),
        ("模型加载", test_model_loading),
        ("特征提取", test_feature_extraction),
        ("Pipeline基础", test_pipeline_basic),
        ("评估指标", test_evaluation),
        ("配置文件", test_config_files),
        ("已训练模型", test_trained_models),
        ("Pipeline配置", test_pipeline_configs),
        ("L2R模型加载", test_l2r_model_loading),
        ("数据格式", test_data_format),
        ("Pipeline检索", test_pipeline_retrieval),
    ]
    
    for test_name, test_func in tests:
        try:
            success, errors = test_func()
            test_results.append((test_name, success, errors))
            all_errors.extend(errors)
        except Exception as e:
            print(f"\n✗ {test_name}测试异常: {e}")
            traceback.print_exc()
            test_results.append((test_name, False, [str(e)]))
            all_errors.append(f"{test_name}: {e}")
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    passed = sum(1 for _, success, _ in test_results if success)
    total = len(test_results)
    
    print(f"\n通过: {passed}/{total}")
    print(f"失败: {total - passed}/{total}")
    
    if all_errors:
        print(f"\n发现的问题 ({len(all_errors)} 个):")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
    else:
        print("\n✓ 所有测试通过！")
    
    print("\n" + "=" * 80)
    
    return len(all_errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

