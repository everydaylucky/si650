from typing import Dict, List
import re

class IRFeatureExtractor:
    """IR特征提取器 (4个特征)"""
    
    def __init__(self):
        self.bm25 = None
        self.tfidf = None
    
    def extract(self, query: Dict, candidate: Dict) -> List[float]:
        """提取IR特征"""
        query_text = query.get("citation_context", "").lower()
        title = candidate.get("title", "").lower()
        abstract = candidate.get("abstract", "").lower()
        
        # 1. BM25 score
        bm25_score = query.get("_bm25_score", 0.0)
        if self.bm25 and bm25_score == 0.0:
            candidate_text = f"{candidate.get('title', '')} {candidate.get('abstract', '')}"
            candidate_id = candidate.get("id") or candidate.get("paper_id")
            results = self.bm25.retrieve(query_text, top_k=1000)
            bm25_score = next((s for pid, s in results if pid == candidate_id), 0.0)
        
        # 2. TF-IDF cosine similarity
        tfidf_score = query.get("_tfidf_score", 0.0)
        if self.tfidf and tfidf_score == 0.0:
            candidate_text = f"{candidate.get('title', '')} {candidate.get('abstract', '')}"
            candidate_id = candidate.get("id") or candidate.get("paper_id")
            results = self.tfidf.retrieve(query_text, top_k=1000)
            tfidf_score = next((s for pid, s in results if pid == candidate_id), 0.0)
        
        # 3. Title term overlap ratio
        query_terms = set(re.findall(r'\w+', query_text))
        title_terms = set(re.findall(r'\w+', title))
        title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
        
        # 4. Abstract term overlap ratio
        abstract_terms = set(re.findall(r'\w+', abstract))
        abstract_overlap = len(query_terms & abstract_terms) / max(len(query_terms), 1)
        
        return [bm25_score, tfidf_score, title_overlap, abstract_overlap]

