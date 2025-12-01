from typing import Dict, List

class CategoryFeatureExtractor:
    """类别特征提取器 (4个特征)"""
    
    def __init__(self):
        # AI/ML核心领域
        self.ai_ml_domains = {
            'cs.LG', 'cs.CV', 'cs.CL', 'cs.AI', 'cs.NE',
            'stat.ML', 'cs.IR', 'cs.MM'
        }
    
    def extract(self, query: Dict, candidate: Dict) -> List[float]:
        """提取类别特征"""
        source_categories = set(query.get("source_categories", []))
        target_categories = set(candidate.get("categories", []))
        
        # 1. Primary category exact match (binary)
        primary_match = 1.0 if source_categories & target_categories else 0.0
        
        # 2. Category overlap ratio (0-1)
        if source_categories:
            overlap_ratio = len(source_categories & target_categories) / len(source_categories)
        else:
            overlap_ratio = 0.0
        
        # 3. Same AI/ML core domain (binary)
        source_ai = bool(source_categories & self.ai_ml_domains)
        target_ai = bool(target_categories & self.ai_ml_domains)
        same_ai_domain = 1.0 if (source_ai and target_ai) else 0.0
        
        # 4. Related category match (binary)
        # 简单判断：如果有任何类别重叠就认为是相关的
        related_match = 1.0 if (source_categories & target_categories) else 0.0
        
        return [primary_match, overlap_ratio, same_ai_domain, related_match]

