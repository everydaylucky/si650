import numpy as np
from typing import List, Tuple

def calculate_mrr(ground_truth: List[str], predictions: List[List[Tuple[str, float]]]) -> float:
    """计算Mean Reciprocal Rank (MRR)"""
    reciprocal_ranks = []
    
    for gt, pred in zip(ground_truth, predictions):
        # 找到ground truth在预测结果中的位置
        rank = None
        for i, (paper_id, _) in enumerate(pred, start=1):
            if paper_id == gt:
                rank = i
                break
        
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def calculate_recall_at_k(ground_truth: List[str], 
                         predictions: List[List[Tuple[str, float]]], 
                         k: int = 10) -> float:
    """计算Recall@K"""
    recalls = []
    
    for gt, pred in zip(ground_truth, predictions):
        top_k_ids = [paper_id for paper_id, _ in pred[:k]]
        if gt in top_k_ids:
            recalls.append(1.0)
        else:
            recalls.append(0.0)
    
    return np.mean(recalls) if recalls else 0.0

def calculate_precision_at_k(ground_truth: List[str],
                           predictions: List[List[Tuple[str, float]]],
                           k: int = 10) -> float:
    """计算Precision@K"""
    precisions = []
    
    for gt, pred in zip(ground_truth, predictions):
        top_k_ids = [paper_id for paper_id, _ in pred[:k]]
        if gt in top_k_ids:
            precisions.append(1.0 / k)
        else:
            precisions.append(0.0)
    
    return np.mean(precisions) if precisions else 0.0

def calculate_ndcg_at_k(ground_truth: List[str],
                       predictions: List[List[Tuple[str, float]]],
                       k: int = 10) -> float:
    """计算NDCG@K"""
    ndcgs = []
    
    for gt, pred in zip(ground_truth, predictions):
        # DCG
        dcg = 0.0
        for i, (paper_id, _) in enumerate(pred[:k], start=1):
            if paper_id == gt:
                rel = 1.0
                dcg += rel / np.log2(i + 1)
                break
        
        # IDCG (ideal case: relevant at position 1)
        idcg = 1.0 / np.log2(2)  # rel=1 at position 1
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    
    return np.mean(ndcgs) if ndcgs else 0.0

