from typing import List, Dict, Tuple
from collections import defaultdict

class ReciprocalRankFusion:
    """Reciprocal Rank Fusion融合器"""
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(self, ranked_lists: List[List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
        """
        融合多个排序列表
        
        Args:
            ranked_lists: List of ranked result lists, each is List[Tuple[paper_id, score]]
            
        Returns:
            Fused ranked list: List[Tuple[paper_id, rrf_score]]
        """
        rrf_scores = defaultdict(float)
        
        for rank_list in ranked_lists:
            for rank, (doc_id, _) in enumerate(rank_list, start=1):
                rrf_scores[doc_id] += 1.0 / (self.k + rank)
        
        # 按分数排序
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def rank(self, 
             query: Dict,
             candidates: List[Dict],
             ranked_lists: List[List[Tuple[str, float]]] = None,
             top_k: int = None) -> List[Tuple[str, float]]:
        """
        对候选文档进行RRF融合排序
        
        Args:
            query: Query dict (not used in RRF, but kept for interface consistency)
            candidates: List of candidate dicts (not used, but kept for interface)
            ranked_lists: List of ranked result lists from different retrievers
            top_k: Number of top results to return
            
        Returns:
            Fused ranked list
        """
        if ranked_lists is None:
            raise ValueError("ranked_lists is required for RRF")
        
        results = self.fuse(ranked_lists)
        
        if top_k:
            return results[:top_k]
        return results

