import numpy as np
from typing import Dict
from .ir_features import IRFeatureExtractor
from .embedding_features import EmbeddingFeatureExtractor
from .category_features import CategoryFeatureExtractor
from .temporal_features import TemporalFeatureExtractor
from .context_features import ContextFeatureExtractor

class FeatureExtractor:
    """特征提取器 (18个特征)"""
    
    def __init__(self):
        self.ir_features = IRFeatureExtractor()
        self.embedding_features = EmbeddingFeatureExtractor()
        self.category_features = CategoryFeatureExtractor()
        self.temporal_features = TemporalFeatureExtractor()
        self.context_features = ContextFeatureExtractor()
    
    def extract_all_features(self, query: Dict, candidate: Dict) -> np.ndarray:
        """提取所有18个特征"""
        features = []
        
        # IR特征 (4个)
        features.extend(self.ir_features.extract(query, candidate))
        
        # 嵌入相似度特征 (4个)
        features.extend(self.embedding_features.extract(query, candidate))
        
        # 类别特征 (4个)
        features.extend(self.category_features.extract(query, candidate))
        
        # 时间特征 (3个)
        features.extend(self.temporal_features.extract(query, candidate))
        
        # 上下文特征 (3个)
        features.extend(self.context_features.extract(query, candidate))
        
        return np.array(features)

