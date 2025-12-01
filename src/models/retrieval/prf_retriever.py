"""
Pseudo-Relevance Feedback (PRF) 查询扩展检索器
"""
from typing import List, Dict, Tuple
from collections import Counter
from src.models.retrieval.bm25_retriever import BM25Retriever
from src.models.base.base_retriever import BaseRetriever

class PRFRetriever(BaseRetriever):
    """伪相关反馈检索器（查询扩展）"""
    
    def __init__(self, 
                 k1: float = 1.5,
                 b: float = 0.75,
                 top_k_initial: int = 5,
                 num_expansion_terms: int = 10,
                 alpha: float = 0.7,
                 beta: float = 0.3):
        self.bm25 = BM25Retriever(k1=k1, b=b)
        self.top_k_initial = top_k_initial
        self.num_expansion_terms = num_expansion_terms
        self.alpha = alpha
        self.beta = beta
    
    def build_index(self, documents: List[Dict]) -> None:
        """构建BM25索引"""
        self.bm25.build_index(documents)
    
    def retrieve(self, query: str, top_k: int = 1000) -> List[Tuple[str, float]]:
        """使用PRF检索"""
        # 步骤1: 初始检索
        initial_results = self.bm25.retrieve(query, top_k=self.top_k_initial)
        
        if not initial_results:
            return []
        
        # 步骤2: 提取扩展词
        expansion_terms = self._extract_expansion_terms(query, initial_results)
        
        # 步骤3: 构建扩展查询
        expanded_query = self._expand_query(query, expansion_terms)
        
        # 步骤4: 使用扩展查询重新检索
        final_results = self.bm25.retrieve(expanded_query, top_k=top_k)
        
        return final_results
    
    def _extract_expansion_terms(self, 
                                 original_query: str,
                                 top_docs: List[Tuple[str, float]]) -> List[str]:
        """从top文档中提取扩展词"""
        # 获取文档文本
        doc_texts = []
        for doc_id, _ in top_docs:
            doc = self.bm25.doc_ids.index(doc_id) if doc_id in self.bm25.doc_ids else None
            if doc is not None and doc < len(self.bm25.tokenized_docs):
                doc_texts.append(self.bm25.tokenized_docs[doc])
        
        # 计算词频
        term_scores = Counter()
        original_terms = set(original_query.lower().split())
        
        for doc_tokens in doc_texts:
            for token in doc_tokens:
                if token not in original_terms and len(token) > 2:
                    term_scores[token] += 1
        
        # 选择top扩展词
        expansion_terms = [term for term, _ in term_scores.most_common(self.num_expansion_terms)]
        return expansion_terms
    
    def _expand_query(self, original_query: str, expansion_terms: List[str]) -> str:
        """构建扩展查询"""
        original_weighted = f"{original_query} " * int(self.alpha * 10)
        expansion_weighted = " ".join(expansion_terms) * int(self.beta * 10)
        return f"{original_weighted} {expansion_weighted}".strip()
    
    def save_index(self, path: str) -> None:
        """保存索引（委托给BM25）"""
        self.bm25.save_index(path)
    
    def load_index(self, path: str) -> None:
        """加载索引（委托给BM25）"""
        self.bm25.load_index(path)

