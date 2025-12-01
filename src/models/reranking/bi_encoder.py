import os
import torch
import traceback
from typing import List, Dict, Tuple
from tqdm import tqdm

# 配置Hugging Face镜像（中国用户）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from sentence_transformers import SentenceTransformer
from src.models.base.base_ranker import BaseRanker

class BiEncoder(BaseRanker):
    """双编码器 (SciBERT)"""
    
    def __init__(self, 
                 model_name: str = "allenai/scibert_scivocab_uncased",
                 device: str = None,
                 fine_tuned_path: str = None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            if fine_tuned_path:
                print(f"正在加载微调模型: {fine_tuned_path}")
                self.model = SentenceTransformer(fine_tuned_path, device=self.device)
            else:
                print(f"正在加载模型: {model_name}")
                self.model = SentenceTransformer(model_name, device=self.device)
            print("✓ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            print(f"   模型路径: {fine_tuned_path or model_name}")
            traceback.print_exc()
            raise
    
    def rank(self, 
             query: Dict,
             candidates: List[Dict],
             top_k: int = None) -> List[Tuple[str, float]]:
        """对候选文档排序"""
        try:
            query_text = query.get("citation_context", "")
            candidate_texts = [
                f"{c.get('title', '')} {c.get('abstract', '')}" for c in candidates
            ]
            
            # 编码
            query_emb = self.model.encode(
                [query_text], 
                convert_to_tensor=True, 
                device=self.device,
                show_progress_bar=len(candidates) > 100
            )
            candidate_embs = self.model.encode(
                candidate_texts, 
                convert_to_tensor=True, 
                device=self.device,
                show_progress_bar=len(candidates) > 100
            )
            
            # 计算相似度
            scores = torch.nn.functional.cosine_similarity(
                query_emb, candidate_embs
            ).cpu().tolist()
            
            # 组合结果
            results = list(zip([c["id"] for c in candidates], scores))
            results.sort(key=lambda x: x[1], reverse=True)
            
            if top_k:
                return results[:top_k]
            return results
        except Exception as e:
            print(f"❌ 排序失败:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            print(f"   候选数量: {len(candidates)}")
            print(f"   查询: {query.get('citation_context', '')[:100]}...")
            traceback.print_exc()
            raise

