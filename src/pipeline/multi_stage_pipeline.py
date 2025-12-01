import traceback
from typing import List, Dict, Tuple
from collections import defaultdict

from src.models.retrieval import BM25Retriever, TFIDFRetriever, DenseRetriever
from src.models.retrieval.prf_retriever import PRFRetriever
from src.models.reranking import ReciprocalRankFusion, BiEncoder
from src.models.ranking import CrossEncoderRanker, L2RRanker

class MultiStagePipeline:
    """多阶段检索管道"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.stage1_retrievers = []
        self.stage2_rerankers = []
        self.stage3_ranker = None
        self.doc_store: Dict[str, Dict] = {}
        self._l2r_needs_init = False
        
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化各阶段模型"""
        stage1_config = self.config.get("stage1", {})
        stage2_config = self.config.get("stage2", {})
        stage3_config = self.config.get("stage3", {})
        
        # Stage 1: 初始检索
        if stage1_config.get("use_bm25", False):
            bm25_config = stage1_config.get("bm25", {})
            self.stage1_retrievers.append(BM25Retriever(
                k1=bm25_config.get("k1", 1.5),
                b=bm25_config.get("b", 0.75)
            ))
        
        if stage1_config.get("use_tfidf", False):
            tfidf_config = stage1_config.get("tfidf", {})
            self.stage1_retrievers.append(TFIDFRetriever(
                max_features=tfidf_config.get("max_features", 10000),
                ngram_range=tuple(tfidf_config.get("ngram_range", [1, 2]))
            ))
        
        if stage1_config.get("use_prf", False):
            prf_config = stage1_config.get("prf", {})
            self.stage1_retrievers.append(PRFRetriever(
                k1=prf_config.get("k1", 1.5),
                b=prf_config.get("b", 0.75),
                top_k_initial=prf_config.get("top_k_initial", 5),
                num_expansion_terms=prf_config.get("num_expansion_terms", 10),
                alpha=prf_config.get("alpha", 0.7),
                beta=prf_config.get("beta", 0.3)
            ))
        
        if stage1_config.get("use_specter2", False):
            specter2_config = stage1_config.get("specter2", {})
            self.stage1_retrievers.append(DenseRetriever(
                model_name=specter2_config.get("model_name", "allenai/specter2"),
                fine_tuned_path=specter2_config.get("fine_tuned_path")
            ))
        
        # Stage 2: 重排序
        if stage2_config.get("use_rrf", False):
            rrf_config = stage2_config.get("rrf", {})
            self.stage2_rerankers.append(ReciprocalRankFusion(
                k=rrf_config.get("k", 60)
            ))
        
        if stage2_config.get("use_bi_encoder", False):
            bi_config = stage2_config.get("bi_encoder", {})
            self.stage2_rerankers.append(BiEncoder(
                model_name=bi_config.get("model_name", "allenai/scibert_scivocab_uncased"),
                fine_tuned_path=bi_config.get("fine_tuned_path")
            ))
        
        # Stage 3: 最终排序
        if stage3_config.get("use_cross_encoder", False):
            ce_config = stage3_config.get("cross_encoder", {})
            self.stage3_ranker = CrossEncoderRanker(
                model_name=ce_config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
                fine_tuned_path=ce_config.get("fine_tuned_path")
            )
        elif stage3_config.get("use_l2r", False):
            l2r_config = stage3_config.get("l2r", {})
            # L2R需要特征提取器，稍后在build_indices后初始化
            self.stage3_ranker = L2RRanker(
                model_path=l2r_config.get("model_path")
            )
            self._l2r_needs_init = True  # 标记需要初始化检索器
    
    def build_indices(self, documents: List[Dict]) -> None:
        """构建所有检索器的索引"""
        if not documents:
            raise ValueError("No documents provided. Cannot build indices.")
        
        # 建立全局文档存储，供后续阶段使用
        self.doc_store = {doc["id"]: doc for doc in documents if "id" in doc}
        
        for i, retriever in enumerate(self.stage1_retrievers):
            try:
                retriever_type = type(retriever).__name__
                print(f"正在构建 {retriever_type} 索引 ({i+1}/{len(self.stage1_retrievers)})...")
                retriever.build_index(documents)
                print(f"✓ {retriever_type} 索引构建完成")
            except Exception as e:
                print(f"❌ {retriever_type} 索引构建失败:")
                print(f"   错误类型: {type(e).__name__}")
                print(f"   错误信息: {str(e)}")
                traceback.print_exc()
                raise
        
        # 如果使用L2R，需要将检索器传递给特征提取器
        if self._l2r_needs_init and self.stage3_ranker and hasattr(self.stage3_ranker, 'feature_extractor'):
            from src.models.retrieval import BM25Retriever, TFIDFRetriever
            # 找到BM25和TF-IDF检索器
            bm25_retriever = None
            tfidf_retriever = None
            for retriever in self.stage1_retrievers:
                if isinstance(retriever, BM25Retriever):
                    bm25_retriever = retriever
                elif isinstance(retriever, TFIDFRetriever):
                    tfidf_retriever = retriever
            
            # 初始化特征提取器
            if self.stage3_ranker.feature_extractor is None:
                from src.features import FeatureExtractor
                self.stage3_ranker.feature_extractor = FeatureExtractor()
            
            # 设置检索器
            if bm25_retriever:
                self.stage3_ranker.feature_extractor.ir_features.bm25 = bm25_retriever
            if tfidf_retriever:
                self.stage3_ranker.feature_extractor.ir_features.tfidf = tfidf_retriever
            
            self._l2r_needs_init = False
    
    def _build_enhanced_query(self, query: Dict) -> str:
        """构建增强的查询文本
        
        支持多种增强模式：
        - Exp 6.1: citation_context + source_paper
        - Exp 6.1b: context_before/after + citation_context
        """
        # 获取上下文增强模式
        context_mode = self.config.get("query_enhancement", {}).get("context_mode", "none")
        # none: 只用 citation_context
        # before: context_before + citation_context
        # after: citation_context + context_after
        # both: context_before + citation_context + context_after
        
        # 提取 citation_context 和前后文
        citation_context = query.get("citation_context", "")
        context_before = query.get("context_before", "")
        context_after = query.get("context_after", "")
        
        # 处理 citation_context 可能是字典的情况
        if isinstance(citation_context, dict):
            query_text = citation_context.get("text", "")
            # 如果 citation_context 是字典，尝试从中提取前后文
            if not context_before:
                context_before = citation_context.get("context_before", "")
            if not context_after:
                context_after = citation_context.get("context_after", "")
        else:
            query_text = str(citation_context)
        
        # 根据模式组合查询
        parts = []
        
        if context_mode == "before":
            if context_before:
                parts.append(context_before.strip())
            parts.append(query_text)
        elif context_mode == "after":
            parts.append(query_text)
            if context_after:
                parts.append(context_after.strip())
        elif context_mode == "both":
            if context_before:
                parts.append(context_before.strip())
            parts.append(query_text)
            if context_after:
                parts.append(context_after.strip())
        else:
            # 默认：只用 citation_context
            parts.append(query_text)
        
        # 检查是否添加 source_paper（如果启用）
        use_source_paper = self.config.get("query_enhancement", {}).get("use_source_paper", False)
        if use_source_paper:
            source_paper = query.get("source_paper", {})
            if not source_paper:
                source_paper_id = query.get("source_paper_id", "")
                if source_paper_id and source_paper_id in self.doc_store:
                    source_paper = self.doc_store[source_paper_id]
            
            if source_paper:
                source_title = source_paper.get("title", "")
                source_abstract = source_paper.get("abstract", "")
                
                if source_title:
                    parts.append(source_title)
                
                max_abstract_length = self.config.get("query_enhancement", {}).get("max_abstract_length", 200)
                if source_abstract:
                    if len(source_abstract) > max_abstract_length:
                        source_abstract = source_abstract[:max_abstract_length] + "..."
                    parts.append(source_abstract)
        
        enhanced_query = " ".join(parts).strip()
        return enhanced_query
    
    def retrieve(self, query: Dict) -> List[Tuple[str, float]]:
        """完整检索流程"""
        if not self.doc_store:
            raise ValueError("No documents indexed. Please call build_indices() first.")
        
        try:
            # 构建增强的查询文本
            query_text = self._build_enhanced_query(query)
            
            stage1_top_k = self.config.get("stage1", {}).get("top_k", 1000)
            stage2_top_k = self.config.get("stage2", {}).get("top_k", 50)
            stage3_top_k = self.config.get("stage3", {}).get("top_k", 20)
            
            # Stage 1: 初始检索
            stage1_results = []
            for retriever in self.stage1_retrievers:
                try:
                    results = retriever.retrieve(query_text, top_k=stage1_top_k)
                    stage1_results.append(results)
                except Exception as e:
                    print(f"❌ Stage1 检索失败 ({type(retriever).__name__}):")
                    print(f"   错误类型: {type(e).__name__}")
                    print(f"   错误信息: {str(e)}")
                    traceback.print_exc()
                    raise
            
            # 合并候选池
            candidate_docs = self._merge_candidates(stage1_results)
            
            if not candidate_docs:
                # 如果没有候选文档，返回空结果
                return []
            
            stage2_candidates = candidate_docs[:stage2_top_k * 2]  # 取更多候选用于重排序
            
            for reranker in self.stage2_rerankers:
                try:
                    if isinstance(reranker, ReciprocalRankFusion):
                        fused_scores = reranker.rank(
                            query,
                            stage2_candidates,
                            ranked_lists=stage1_results,
                            top_k=stage2_top_k,
                        )
                        stage2_candidates = self._docs_from_score_list(fused_scores)
                    else:
                        reranked = reranker.rank(query, stage2_candidates, top_k=stage2_top_k)
                        stage2_candidates = self._docs_from_score_list(reranked)
                except Exception as e:
                    print(f"❌ Stage2 重排序失败 ({type(reranker).__name__}):")
                    print(f"   错误类型: {type(e).__name__}")
                    print(f"   错误信息: {str(e)}")
                    traceback.print_exc()
                    raise
            
            # Stage 3: 最终排序
            if self.stage3_ranker:
                try:
                    # 确保有候选文档
                    if not stage2_candidates:
                        # 如果没有Stage2候选，使用Stage1的结果
                        stage2_candidates = candidate_docs[:stage3_top_k * 2]
                    
                    if not stage2_candidates:
                        # 如果仍然没有候选，返回空结果
                        return []
                    
                    stage3_candidates = stage2_candidates[:stage3_top_k * 2]
                    
                    # 确保query格式正确（citation_context是字符串）
                    query_for_l2r = query.copy()
                    if isinstance(query_for_l2r.get("citation_context"), dict):
                        query_for_l2r["citation_context"] = query_for_l2r["citation_context"].get("text", "")
                    elif not query_for_l2r.get("citation_context"):
                        # 如果citation_context为空，使用query_text
                        query_for_l2r["citation_context"] = query_text
                    
                    final_results = self.stage3_ranker.rank(query_for_l2r, stage3_candidates, top_k=stage3_top_k)
                except Exception as e:
                    print(f"❌ Stage3 排序失败 ({type(self.stage3_ranker).__name__}):")
                    print(f"   错误类型: {type(e).__name__}")
                    print(f"   错误信息: {str(e)}")
                    traceback.print_exc()
                    raise
            else:
                final_results = [
                    (doc["id"], 0.0)
                    for doc in stage2_candidates[:stage3_top_k]
                ]
            
            return final_results
        except Exception as e:
            print(f"❌ 检索流程失败:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            print(f"   查询: {query.get('citation_context', '')[:100]}...")
            traceback.print_exc()
            raise
    
    def _merge_candidates(self, result_lists: List[List[Tuple[str, float]]]) -> List[Dict]:
        """合并多个检索结果，并保留文档元信息"""
        candidate_scores = defaultdict(float)
        for result_list in result_lists:
            for paper_id, score in result_list:
                candidate_scores[paper_id] = max(candidate_scores[paper_id], score)
        
        # 按得分排序并映射到文档
        sorted_ids = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = []
        for paper_id, _ in sorted_ids:
            doc = self.doc_store.get(paper_id)
            if doc:
                candidates.append(doc)
        return candidates
    
    def _docs_from_score_list(self, score_list: List[Tuple[str, float]]) -> List[Dict]:
        """根据(文档ID, 分数)列表返回对应的文档信息，按分数排序"""
        docs = []
        for paper_id, _ in score_list:
            doc = self.doc_store.get(paper_id)
            if doc:
                docs.append(doc)
        return docs

