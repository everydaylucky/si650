import os
import json
import traceback
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# 配置Hugging Face镜像
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

class SciBERTTrainer:
    """SciBERT模型训练器"""
    
    def __init__(self, 
                 model_name: str = "allenai/scibert_scivocab_uncased",
                 device: str = None):
        self.model_name = model_name
        # 修复设备检测逻辑
        if device:
            self.device = device
        else:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
    def load_data(self, train_file: str, val_file: str = None) -> Tuple[List[InputExample], List[InputExample]]:
        """加载训练数据"""
        print("=" * 60)
        print("加载训练数据...")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        print(f"训练样本数: {len(train_data)}")
        
        # 转换为InputExample
        train_examples = []
        for sample in tqdm(train_data, desc="处理训练样本"):
            query = sample.get("citation_context", "")
            positive = self._format_paper(sample.get("target_paper", {}))
            negatives = sample.get("negatives", []) or []  # 确保是列表
            
            if not query or not positive:
                continue
            
            # 正样本对
            train_examples.append(InputExample(
                texts=[query, positive],
                label=1.0
            ))
            
            # 负样本对
            if isinstance(negatives, list):
                for neg in negatives[:5]:  # 限制负样本数量
                    if isinstance(neg, dict):
                        neg_text = self._format_paper(neg)
                        if neg_text:
                            train_examples.append(InputExample(
                                texts=[query, neg_text],
                                label=0.0
                            ))
        
        print(f"✓ 训练样本对: {len(train_examples)}")
        
        # 验证集
        val_examples = []
        if val_file and Path(val_file).exists():
            print(f"\n加载验证数据: {val_file}")
            with open(val_file, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
            
            print(f"验证样本数: {len(val_data)}")
            
            for sample in tqdm(val_data, desc="处理验证样本"):
                query = sample.get("citation_context", "")
                positive = self._format_paper(sample.get("target_paper", {}))
                negatives = sample.get("negatives", []) or []  # 确保是列表
                
                if not query or not positive:
                    continue
                
                # 创建评估样本（查询 + 正样本 + 负样本）
                corpus = {positive: 0}  # 正样本label=0（最高）
                queries = {query: [positive]}
                
                if isinstance(negatives, list):
                    for i, neg in enumerate(negatives[:20], 1):  # 限制负样本
                        if isinstance(neg, dict):
                            neg_text = self._format_paper(neg)
                            if neg_text:
                                corpus[neg_text] = i
                                queries[query].append(neg_text)
                
                if len(queries[query]) > 1:
                    val_examples.append({
                        'query': query,
                        'corpus': corpus,
                        'queries': queries
                    })
            
            print(f"✓ 验证样本对: {len(val_examples)}")
        
        return train_examples, val_examples
    
    def _format_paper(self, paper: Dict) -> str:
        """格式化论文为文本"""
        if not paper:
            return ""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        return f"{title} {abstract}".strip()
    
    def train(self,
              train_file: str,
              val_file: str = None,
              output_dir: str = "experiments/checkpoints/scibert",
              epochs: int = 3,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              warmup_steps: int = 100,
              early_stopping_patience: int = 2):
        """训练模型"""
        try:
            print("=" * 60)
            print("开始训练SciBERT模型")
            print(f"模型: {self.model_name}")
            print(f"设备: {self.device}")
            print(f"训练轮次: {epochs}")
            print(f"批次大小: {batch_size}")
            print(f"学习率: {learning_rate}")
            print("=" * 60)
            
            # 加载模型
            print("\n加载预训练模型...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print("✓ 模型加载成功")
            
            # 加载数据
            train_examples, val_examples = self.load_data(train_file, val_file)
            
            if not train_examples:
                raise ValueError("没有可用的训练样本")
            
            # 创建数据加载器
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=batch_size
            )
            
            # 定义损失函数（MultipleNegativesRankingLoss）
            train_loss = losses.MultipleNegativesRankingLoss(self.model)
            
            # 验证评估器（简化版：使用CosineSimilarityEvaluator）
            evaluator = None
            if val_examples:
                print("\n创建验证评估器...")
                # 使用部分验证样本创建简单的相似度评估
                eval_samples = val_examples[:min(50, len(val_examples))]
                eval_pairs = []
                eval_labels = []
                
                for ex in eval_samples:
                    query = ex['query']
                    # 正样本
                    positive = [k for k, v in ex['corpus'].items() if v == 0]
                    if positive:
                        eval_pairs.append(InputExample(texts=[query, positive[0]], label=1.0))
                        eval_labels.append(1.0)
                    # 负样本（取前几个）
                    negatives = [k for k, v in ex['corpus'].items() if v > 0][:3]
                    for neg in negatives:
                        eval_pairs.append(InputExample(texts=[query, neg], label=0.0))
                        eval_labels.append(0.0)
                
                if eval_pairs:
                    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
                        eval_pairs,
                        name='validation',
                        show_progress_bar=True
                    )
                    print(f"✓ 验证评估器创建成功 ({len(eval_pairs)} 样本对)")
            
            # 训练
            print("\n" + "=" * 60)
            print("开始训练...")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                optimizer_params={'lr': learning_rate},
                evaluator=evaluator,
                evaluation_steps=500 if evaluator else None,
                output_path=str(output_path),
                save_best_model=True,
                show_progress_bar=True
            )
            
            print("\n" + "=" * 60)
            print(f"✓ 训练完成！模型已保存到: {output_path}")
            print("=" * 60)
            
            return str(output_path)
            
        except Exception as e:
            print(f"\n❌ 训练失败:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            traceback.print_exc()
            raise

