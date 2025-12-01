import os
import torch
from typing import List, Dict, Tuple

# 配置Hugging Face镜像（中国用户）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from sentence_transformers import CrossEncoder
from src.models.base.base_ranker import BaseRanker

class CrossEncoderRanker(BaseRanker):
    """Cross-Encoder排序器"""
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
                 fine_tuned_path: str = None,
                 device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        if fine_tuned_path:
            self.model = CrossEncoder(fine_tuned_path, device=self.device)
        else:
            self.model = CrossEncoder(model_name, device=self.device)
    
    def rank(self, 
             query: Dict,
             candidates: List[Dict],
             top_k: int = None) -> List[Tuple[str, float]]:
        """Cross-Encoder排序"""
        query_text = query.get("citation_context", "")
        
        # 构建query-doc对
        pairs = [
            [query_text, f"{c.get('title', '')} {c.get('abstract', '')}"]
            for c in candidates
        ]
        
        # 批量预测
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=True)
        
        # 组合结果
        results = list(zip([c["id"] for c in candidates], scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            return results[:top_k]
        return results

