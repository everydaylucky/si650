from typing import Dict, List

class EmbeddingFeatureExtractor:
    """嵌入相似度特征提取器 (4个特征)"""
    
    def __init__(self):
        pass
    
    def extract(self, query: Dict, candidate: Dict) -> List[float]:
        """提取嵌入相似度特征
        
        注意: 这些特征需要从实际的检索器/排序器中获取
        这里返回占位符值
        """
        # 1. SPECTER2 cosine similarity
        specter2_sim = query.get("_specter2_score", 0.0)
        
        # 2. SciBERT cosine similarity
        scibert_sim = query.get("_scibert_score", 0.0)
        
        # 3. ColBERT MaxSim score
        colbert_score = query.get("_colbert_score", 0.0)
        
        # 4. Cross-Encoder score
        cross_encoder_score = query.get("_cross_encoder_score", 0.0)
        
        return [specter2_sim, scibert_sim, colbert_score, cross_encoder_score]

