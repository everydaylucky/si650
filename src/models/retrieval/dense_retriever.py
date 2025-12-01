import os
import pickle
import faiss
import numpy as np
import torch
import traceback
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# 配置Hugging Face镜像（中国用户）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from transformers import AutoModel, AutoTokenizer
from src.models.base.base_retriever import BaseRetriever

class DenseRetriever(BaseRetriever):
    """密集检索器 (SPECTER2)"""
    
    def __init__(self, model_name: str = "allenai/specter2", fine_tuned_path: str = None, device: str = None):
        self.model_name = model_name
        self.fine_tuned_path = fine_tuned_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.faiss_index = None
        self.doc_ids = []
        self.embedding_dim = 768
        self.use_sentence_transformer = False  # 标记是否使用SentenceTransformer
    
    def build_index(self, documents: List[Dict]) -> None:
        """构建FAISS索引和嵌入"""
        try:
            # 优先使用fine-tuned模型
            self.use_sentence_transformer = False
            if self.fine_tuned_path and Path(self.fine_tuned_path).exists():
                print(f"正在加载fine-tuned模型: {self.fine_tuned_path}")
                try:
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(self.fine_tuned_path, device=self.device)
                    self.use_sentence_transformer = True
                    # SentenceTransformer的嵌入维度
                    test_embedding = self.model.encode("test", convert_to_numpy=True)
                    self.embedding_dim = test_embedding.shape[-1]
                    print(f"✓ Fine-tuned模型加载成功 (嵌入维度: {self.embedding_dim})")
                except Exception as e:
                    print(f"⚠ Fine-tuned模型加载失败: {e}")
                    print(f"   回退到预训练模型...")
                    self._load_pretrained_model()
            else:
                self._load_pretrained_model()
            
            if not self.use_sentence_transformer:
                self.model.eval()
                # 获取嵌入维度（仅对transformers模型）
                test_input = self.tokenizer("test", return_tensors="pt").to(self.device)
                with torch.no_grad():
                    test_output = self.model(**test_input)
                    if hasattr(test_output, 'last_hidden_state'):
                        self.embedding_dim = test_output.last_hidden_state.shape[-1]
                    elif hasattr(test_output, 'pooler_output'):
                        self.embedding_dim = test_output.pooler_output.shape[-1]
                    else:
                        self.embedding_dim = 768  # 默认值
                print(f"✓ 模型加载成功 (嵌入维度: {self.embedding_dim})")
        except Exception as e:
            print(f"❌ 模型加载失败:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            traceback.print_exc()
            raise
        
        self.doc_ids = [doc["id"] for doc in documents]
        texts = [f"{doc.get('title', '')} {doc.get('abstract', '')}" for doc in documents]
        
        # 批量编码
        print(f"正在编码 {len(texts)} 个文档...")
        try:
            embeddings = self._encode_batch(texts)
            print("✓ 文档编码完成")
        except Exception as e:
            print(f"❌ 文档编码失败:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            traceback.print_exc()
            raise
        
        # 构建FAISS索引
        print("正在构建FAISS索引...")
        try:
            num_docs = len(embeddings)
            nlist = min(100, max(1, num_docs // 10))  # 动态调整聚类数
            
            if num_docs < nlist:
                # 文档数少于聚类数，使用Flat索引
                print(f"⚠ 文档数({num_docs})少于聚类数({nlist})，使用Flat索引")
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            else:
                # 使用IVF索引
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                self.faiss_index.train(embeddings.astype('float32'))
                self.faiss_index.nprobe = min(10, nlist)
            
            self.faiss_index.add(embeddings.astype('float32'))
            print("✓ FAISS索引构建完成")
        except Exception as e:
            print(f"❌ FAISS索引构建失败:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            traceback.print_exc()
            raise
    
    def _load_pretrained_model(self):
        """加载预训练模型"""
        print(f"正在加载预训练模型: {self.model_name}")
        # 尝试使用基础模型名称，如果失败则使用原始名称
        base_model_name = "allenai/specter2_base"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = AutoModel.from_pretrained(base_model_name).to(self.device)
            print(f"✓ 使用基础模型: {base_model_name}")
        except Exception:
            # 如果基础模型不存在，尝试直接加载原始模型（可能需要特殊处理）
            print(f"⚠ 基础模型加载失败，尝试直接加载: {self.model_name}")
            # 尝试使用 transformers 直接加载，忽略 adapter 配置
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                # 尝试获取基础模型名称
                if hasattr(config, 'base_model_name_or_path'):
                    base_name = config.base_model_name_or_path
                    print(f"   检测到基础模型: {base_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(base_name)
                    self.model = AutoModel.from_pretrained(base_name).to(self.device)
                else:
                    # 最后尝试：使用 scibert 作为替代
                    print(f"   使用替代模型: allenai/scibert_scivocab_uncased")
                    self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
                    self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(self.device)
            except Exception as e2:
                print(f"   替代方案也失败，使用 SciBERT: {str(e2)}")
                self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
                self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(self.device)
    
    def retrieve(self, query: str, top_k: int = 1000) -> List[Tuple[str, float]]:
        """检索文档"""
        if self.faiss_index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        try:
            query_embedding = self._encode([query])
            distances, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.doc_ids))
            )
            
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.doc_ids):
                    score = 1.0 / (1.0 + dist)  # 转换为相似度分数
                    results.append((self.doc_ids[idx], score))
            return results
        except Exception as e:
            print(f"❌ 检索失败:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            print(f"   查询: {query[:100]}...")
            traceback.print_exc()
            raise
    
    def save_index(self, path: str) -> None:
        """保存索引"""
        index_data = {
            'faiss_index': self.faiss_index,
            'doc_ids': self.doc_ids,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        }
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
    
    def load_index(self, path: str) -> None:
        """加载索引"""
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        self.faiss_index = index_data['faiss_index']
        self.doc_ids = index_data['doc_ids']
        self.model_name = index_data['model_name']
        self.embedding_dim = index_data['embedding_dim']
        
        # 重新加载模型
        try:
            base_model_name = "allenai/specter2_base"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = AutoModel.from_pretrained(base_model_name).to(self.device)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
            self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(self.device)
        self.model.eval()
    
    def _encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码"""
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), desc="编码批次", unit="batch", total=total_batches):
            try:
                batch = texts[i:i+batch_size]
                batch_embeddings = self._encode(batch)
                embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"\n❌ 批次 {i//batch_size + 1}/{total_batches} 编码失败:")
                print(f"   错误类型: {type(e).__name__}")
                print(f"   错误信息: {str(e)}")
                print(f"   批次范围: {i} - {min(i+batch_size, len(texts))}")
                traceback.print_exc()
                raise
        return np.vstack(embeddings)
    
    def _encode(self, texts: List[str]) -> np.ndarray:
        """编码文本"""
        # 如果使用sentence-transformers模型（fine-tuned），直接使用其encode方法
        if self.use_sentence_transformer:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        else:
            # 使用 transformers 编码
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用 [CLS] token 的嵌入（第一个token）
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output.cpu().numpy()
                else:
                    # 如果都没有，尝试获取第一个隐藏状态
                    embeddings = list(outputs.values())[0][:, 0, :].cpu().numpy()
        
        # L2归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        embeddings = embeddings / norms
        
        return embeddings

