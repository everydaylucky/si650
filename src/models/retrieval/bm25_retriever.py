import pickle
import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from src.models.base.base_retriever import BaseRetriever

class BM25Retriever(BaseRetriever):
    """BM25检索器"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.doc_ids = []
        self.tokenized_docs = []
    
    def build_index(self, documents: List[Dict]) -> None:
        """构建BM25索引"""
        self.doc_ids = [doc["id"] for doc in documents]
        texts = [f"{doc.get('title', '')} {doc.get('abstract', '')}" for doc in documents]
        self.tokenized_docs = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
    
    def retrieve(self, query: str, top_k: int = 1000) -> List[Tuple[str, float]]:
        """检索文档"""
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 转换为numpy数组以便排序
        scores = np.array(scores)
        
        # 获取top_k结果
        top_indices = scores.argsort()[-top_k:][::-1]
        results = [(self.doc_ids[i], float(scores[i])) for i in top_indices]
        return results
    
    def save_index(self, path: str) -> None:
        """保存索引"""
        index_data = {
            'bm25': self.bm25,
            'doc_ids': self.doc_ids,
            'tokenized_docs': self.tokenized_docs,
            'k1': self.k1,
            'b': self.b
        }
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
    
    def load_index(self, path: str) -> None:
        """加载索引"""
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        self.bm25 = index_data['bm25']
        self.doc_ids = index_data['doc_ids']
        self.tokenized_docs = index_data['tokenized_docs']
        self.k1 = index_data['k1']
        self.b = index_data['b']
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        return text.lower().split()

