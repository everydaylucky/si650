from typing import Dict, List
import re

class ContextFeatureExtractor:
    """上下文特征提取器 (3个特征)"""
    
    def __init__(self):
        pass
    
    def extract(self, query: Dict, candidate: Dict) -> List[float]:
        """提取上下文特征"""
        citation_text = query.get("citation_context", "")
        abstract = candidate.get("abstract", "")
        
        # 1. Citation text length (words)
        citation_length = len(re.findall(r'\w+', citation_text))
        
        # 2. Abstract length ratio
        abstract_length = len(re.findall(r'\w+', abstract))
        # 归一化到0-1 (假设最大长度为2000)
        abstract_ratio = min(abstract_length / 2000.0, 1.0)
        
        # 3. Number of categories
        num_categories = float(len(candidate.get("categories", [])))
        
        return [citation_length, abstract_ratio, num_categories]

