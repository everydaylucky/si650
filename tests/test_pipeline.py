import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import MultiStagePipeline

class TestMultiStagePipeline(unittest.TestCase):
    """测试多阶段管道"""
    
    def setUp(self):
        self.config = {
            "stage1": {
                "use_bm25": True,
                "use_tfidf": False,
                "use_specter2": False,
                "top_k": 10
            },
            "stage2": {
                "use_rrf": False,
                "use_bi_encoder": False,
                "top_k": 5
            },
            "stage3": {
                "use_cross_encoder": False,
                "use_l2r": False,
                "top_k": 3
            }
        }
        
        self.documents = [
            {
                "id": "1",
                "title": "Machine Learning",
                "abstract": "Deep learning is a subset of machine learning"
            },
            {
                "id": "2",
                "title": "Natural Language Processing",
                "abstract": "NLP deals with language understanding"
            },
            {
                "id": "3",
                "title": "Computer Vision",
                "abstract": "CV focuses on image processing"
            }
        ]
    
    def test_initialization(self):
        """测试管道初始化"""
        pipeline = MultiStagePipeline(self.config)
        self.assertEqual(len(pipeline.stage1_retrievers), 1)  # 只有BM25
    
    def test_build_indices(self):
        """测试索引构建"""
        pipeline = MultiStagePipeline(self.config)
        pipeline.build_indices(self.documents)
        
        # 检查索引是否构建
        for retriever in pipeline.stage1_retrievers:
            self.assertIsNotNone(retriever.bm25)
    
    def test_retrieve(self):
        """测试检索流程"""
        pipeline = MultiStagePipeline(self.config)
        pipeline.build_indices(self.documents)
        
        query = {
            "citation_context": "machine learning techniques",
            "source_paper_id": "source1"
        }
        
        results = pipeline.retrieve(query)
        
        # 应该有结果
        self.assertGreater(len(results), 0)
        # 结果格式应该是 (paper_id, score)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 2)

if __name__ == '__main__':
    unittest.main()

