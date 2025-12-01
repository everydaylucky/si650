import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.retrieval import BM25Retriever, TFIDFRetriever

class TestBM25Retriever(unittest.TestCase):
    """测试BM25检索器"""
    
    def setUp(self):
        self.retriever = BM25Retriever(k1=1.5, b=0.75)
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
        self.retriever.build_index(self.documents)
    
    def test_build_index(self):
        """测试索引构建"""
        self.assertIsNotNone(self.retriever.bm25)
        self.assertEqual(len(self.retriever.doc_ids), 3)
    
    def test_retrieve(self):
        """测试检索"""
        results = self.retriever.retrieve("machine learning", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "1")  # 应该返回最相关的文档
    
    def test_save_load_index(self):
        """测试保存和加载索引"""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            self.retriever.save_index(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # 创建新的retriever并加载
            new_retriever = BM25Retriever()
            new_retriever.load_index(temp_path)
            
            # 测试加载后的检索
            results = new_retriever.retrieve("machine learning", top_k=1)
            self.assertEqual(len(results), 1)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

class TestTFIDFRetriever(unittest.TestCase):
    """测试TF-IDF检索器"""
    
    def setUp(self):
        self.retriever = TFIDFRetriever(max_features=100, ngram_range=(1, 1))
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
            }
        ]
        self.retriever.build_index(self.documents)
    
    def test_build_index(self):
        """测试索引构建"""
        self.assertIsNotNone(self.retriever.doc_vectors)
        self.assertEqual(len(self.retriever.doc_ids), 2)
    
    def test_retrieve(self):
        """测试检索"""
        results = self.retriever.retrieve("machine learning", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "1")

if __name__ == '__main__':
    unittest.main()

