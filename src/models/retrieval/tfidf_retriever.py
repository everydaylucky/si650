import pickle
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.models.base.base_retriever import BaseRetriever

class TFIDFRetriever(BaseRetriever):
    """TF-IDF检索器"""
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.8,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.base_min_df = min_df
        self.base_max_df = max_df
        self.vectorizer = None
        self.doc_vectors = None
        self.doc_ids = []
    
    def build_index(self, documents: List[Dict]) -> None:
        """构建TF-IDF索引"""
        if not documents:
            raise ValueError("No documents provided for TF-IDF index.")
        
        self.doc_ids = [doc["id"] for doc in documents]
        texts = [f"{doc.get('title', '')} {doc.get('abstract', '')}" for doc in documents]
        doc_count = len(texts)
        self.vectorizer = self._create_vectorizer(doc_count)
        self.doc_vectors = self.vectorizer.fit_transform(texts)
    
    def retrieve(self, query: str, top_k: int = 1000) -> List[Tuple[str, float]]:
        """检索文档"""
        if self.doc_vectors is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = [(self.doc_ids[i], float(similarities[i])) for i in top_indices]
        return results
    
    def save_index(self, path: str) -> None:
        """保存索引"""
        index_data = {
            'vectorizer': self.vectorizer,
            'doc_vectors': self.doc_vectors,
            'doc_ids': self.doc_ids,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'base_min_df': self.base_min_df,
            'base_max_df': self.base_max_df,
        }
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
    
    def load_index(self, path: str) -> None:
        """加载索引"""
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        self.vectorizer = index_data['vectorizer']
        self.doc_vectors = index_data['doc_vectors']
        self.doc_ids = index_data['doc_ids']
        self.max_features = index_data.get('max_features', self.max_features)
        self.ngram_range = index_data.get('ngram_range', self.ngram_range)
        self.base_min_df = index_data.get('base_min_df', self.base_min_df)
        self.base_max_df = index_data.get('base_max_df', self.base_max_df)
    
    def _create_vectorizer(self, doc_count: int) -> TfidfVectorizer:
        """根据语料大小动态调整min_df和max_df，避免sklearn报错"""
        if doc_count <= self.base_min_df:
            min_df = 1
        else:
            min_df = self.base_min_df
        
        max_df = self.base_max_df
        if isinstance(max_df, float):
            # 当max_df对应的文档数小于min_df时，提升到1.0
            if (max_df * doc_count) < min_df:
                max_df = 1.0
            else:
                max_df = min(1.0, max_df)
        elif isinstance(max_df, int):
            max_df = min(max_df, doc_count)
            if max_df < min_df:
                max_df = min_df
        
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=min_df,
            max_df=max_df,
            norm='l2'
        )

