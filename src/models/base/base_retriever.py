from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class BaseRetriever(ABC):
    """检索器基类"""
    
    @abstractmethod
    def build_index(self, documents: List[Dict]) -> None:
        """构建索引
        
        Args:
            documents: List of dicts with keys ['id', 'title', 'abstract', ...]
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 1000) -> List[Tuple[str, float]]:
        """检索文档
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of (paper_id, score) tuples, sorted by score descending
        """
        pass
    
    @abstractmethod
    def save_index(self, path: str) -> None:
        """保存索引到文件"""
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> None:
        """从文件加载索引"""
        pass

