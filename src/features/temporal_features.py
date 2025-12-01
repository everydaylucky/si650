from typing import Dict, List

class TemporalFeatureExtractor:
    """时间特征提取器 (3个特征)"""
    
    def __init__(self, current_year: int = 2024):
        self.current_year = current_year
    
    def extract(self, query: Dict, candidate: Dict) -> List[float]:
        """提取时间特征"""
        source_year = query.get("source_year", self.current_year)
        target_year = candidate.get("year", self.current_year)
        
        # 1. Year difference (source - target)
        year_diff = float(source_year - target_year)
        
        # 2. Target paper recency (2024 - target)
        recency = float(self.current_year - target_year)
        
        # 3. Is recent (<2 years, binary)
        is_recent = 1.0 if recency < 2 else 0.0
        
        return [year_diff, recency, is_recent]

