from typing import List, Dict
from .metrics import calculate_mrr, calculate_recall_at_k, calculate_ndcg_at_k, calculate_precision_at_k

class Evaluator:
    """评估器"""
    
    def __init__(self):
        self.metrics = {
            "mrr": calculate_mrr,
            "recall@5": lambda y_true, y_pred: calculate_recall_at_k(y_true, y_pred, k=5),
            "recall@10": lambda y_true, y_pred: calculate_recall_at_k(y_true, y_pred, k=10),
            "recall@20": lambda y_true, y_pred: calculate_recall_at_k(y_true, y_pred, k=20),
            "recall@50": lambda y_true, y_pred: calculate_recall_at_k(y_true, y_pred, k=50),
            "precision@10": lambda y_true, y_pred: calculate_precision_at_k(y_true, y_pred, k=10),
            "precision@20": lambda y_true, y_pred: calculate_precision_at_k(y_true, y_pred, k=20),
            "ndcg@10": lambda y_true, y_pred: calculate_ndcg_at_k(y_true, y_pred, k=10),
            "ndcg@20": lambda y_true, y_pred: calculate_ndcg_at_k(y_true, y_pred, k=20),
        }
    
    def evaluate(self, 
                 predictions: List[List[tuple]],
                 ground_truth: List[str]) -> Dict[str, float]:
        """评估模型性能
        
        Args:
            predictions: List of ranked result lists, each is List[Tuple[paper_id, score]]
            ground_truth: List of ground truth paper IDs
            
        Returns:
            Dictionary of metric names and values
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Predictions ({len(predictions)}) and ground truth ({len(ground_truth)}) must have same length")
        
        results = {}
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(ground_truth, predictions)
        
        return results

