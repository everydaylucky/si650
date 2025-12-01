import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation import Evaluator, calculate_mrr, calculate_recall_at_k

class TestEvaluation(unittest.TestCase):
    """测试评估指标"""
    
    def test_calculate_mrr(self):
        """测试MRR计算"""
        ground_truth = ["1", "2", "3"]
        predictions = [
            [("1", 0.9), ("4", 0.8)],  # GT在位置1
            [("5", 0.9), ("2", 0.8)],  # GT在位置2
            [("6", 0.9), ("7", 0.8)]   # GT不在top-2
        ]
        
        mrr = calculate_mrr(ground_truth, predictions)
        self.assertAlmostEqual(mrr, (1.0 + 0.5 + 0.0) / 3, places=2)
    
    def test_calculate_recall_at_k(self):
        """测试Recall@K计算"""
        ground_truth = ["1", "2", "3"]
        predictions = [
            [("1", 0.9), ("4", 0.8)],  # GT在top-2
            [("5", 0.9), ("2", 0.8)],  # GT在top-2
            [("6", 0.9), ("7", 0.8)]   # GT不在top-2
        ]
        
        recall = calculate_recall_at_k(ground_truth, predictions, k=2)
        self.assertAlmostEqual(recall, (1.0 + 1.0 + 0.0) / 3, places=2)
    
    def test_evaluator(self):
        """测试评估器"""
        evaluator = Evaluator()
        
        ground_truth = ["1", "2"]
        predictions = [
            [("1", 0.9), ("4", 0.8)],
            [("2", 0.9), ("5", 0.8)]
        ]
        
        results = evaluator.evaluate(predictions, ground_truth)
        
        self.assertIn("mrr", results)
        self.assertIn("recall@10", results)
        self.assertIn("ndcg@10", results)
        self.assertGreater(results["mrr"], 0)

if __name__ == '__main__':
    unittest.main()

