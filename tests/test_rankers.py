import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.reranking import ReciprocalRankFusion

class TestReciprocalRankFusion(unittest.TestCase):
    """测试RRF融合器"""
    
    def setUp(self):
        self.rrf = ReciprocalRankFusion(k=60)
    
    def test_fuse(self):
        """测试RRF融合"""
        ranked_lists = [
            [("1", 0.9), ("2", 0.8), ("3", 0.7)],  # System 1
            [("2", 0.9), ("1", 0.8), ("3", 0.7)],  # System 2
            [("3", 0.9), ("1", 0.8), ("2", 0.7)]   # System 3
        ]
        
        results = self.rrf.fuse(ranked_lists)
        
        # 应该有3个结果
        self.assertEqual(len(results), 3)
        
        # 检查结果格式
        for paper_id, score in results:
            self.assertIsInstance(paper_id, str)
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0)

if __name__ == '__main__':
    unittest.main()

