import lightgbm as lgb
import numpy as np
from typing import List, Dict, Tuple
from src.models.base.base_ranker import BaseRanker

try:
    from src.features import FeatureExtractor
except ImportError:
    FeatureExtractor = None

class L2RRanker(BaseRanker):
    """LightGBM Learning-to-Rank排序器"""
    
    def __init__(self, model_path: str = None, feature_extractor=None):
        self.model = None
        self.feature_extractor = feature_extractor
        
        if model_path:
            try:
                self.model = lgb.Booster(model_file=model_path)
            except Exception as e:
                raise ValueError(f"Failed to load L2R model from {model_path}: {str(e)}")
        elif FeatureExtractor is None:
            raise ValueError("FeatureExtractor is required for L2R. Please implement it first.")
    
    def rank(self, 
             query: Dict,
             candidates: List[Dict],
             top_k: int = None) -> List[Tuple[str, float]]:
        """L2R排序"""
        if self.model is None:
            raise ValueError("Model not loaded. Please provide model_path or train a model first.")
        
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()
        
        # 提取特征
        features = []
        for candidate in candidates:
            feat = self.feature_extractor.extract_all_features(query, candidate)
            # 检查特征是否为空
            if len(feat) == 0:
                raise ValueError(f"特征提取失败：返回空特征数组。查询: {query.get('citation_context', '')[:50]}...")
            features.append(feat)
        
        features = np.array(features)
        
        # 检查特征维度
        if features.size == 0:
            raise ValueError(f"特征数组为空。候选数量: {len(candidates)}")
        
        # 确保是2维数组（LightGBM要求）
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 检查特征数量
        if features.shape[1] != 18:
            raise ValueError(f"特征数量不匹配：期望18个，实际{features.shape[1]}个")
        
        # 预测
        scores = self.model.predict(features)
        
        # 组合结果
        results = list(zip([c["id"] for c in candidates], scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            return results[:top_k]
        return results

