from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class BaseRanker(ABC):
    """排序器基类"""
    
    @abstractmethod
    def rank(self, 
             query: Dict,
             candidates: List[Dict],
             top_k: int = None) -> List[Tuple[str, float]]:
        """对候选文档进行排序
        
        Args:
            query: Query dict with keys like 'citation_context', 'source_paper_id', etc.
            candidates: List of candidate dicts with keys ['id', 'title', 'abstract', ...]
            top_k: Number of top results to return (None for all)
            
        Returns:
            List of (paper_id, score) tuples, sorted by score descending
        """
        pass

