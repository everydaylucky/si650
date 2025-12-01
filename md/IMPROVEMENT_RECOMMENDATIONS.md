# Citation Recommendation 性能提升建议

## 📊 当前性能分析

### 当前最佳结果
- **Multi-Stage Pipeline Optimized**: MRR = 0.3428, R@10 = 0.5996, NDCG@10 = 0.3977
- **L2R Fine-tuned**: MRR = 0.2922, R@10 = 0.5148
- **SPECTER2 Fine-tuned**: MRR = 0.3173, R@10 = 0.5212

### 性能瓶颈分析

1. **查询表示不够丰富**
   - 当前只使用 `citation_context` 文本
   - 没有利用 source paper 的 title/abstract
   - 没有利用 section 信息

2. **特征工程有限**
   - L2R 只有 18 个特征
   - 缺少一些高级特征（如引用网络、作者信息等）

3. **负样本质量**
   - 训练时只用了 20 个负样本
   - 可能缺少 hard negatives

4. **模型融合策略**
   - RRF 参数可能未优化
   - 没有尝试加权融合

---

## 🚀 改进建议（按优先级排序）

### 1. 增强查询表示 ⭐⭐⭐⭐⭐

**问题**: 当前只使用 citation context，信息有限

**改进方案**:

#### A. 多字段查询增强
```python
# 当前: 只用 citation_context
query_text = citation_context

# 改进: 组合多个字段
query_text = f"{citation_context} {source_paper_title} {source_paper_abstract[:200]}"
```

#### B. 上下文扩展
```python
# 添加前后文
if "context_before" in query:
    query_text = f"{context_before} {citation_context} {context_after}"
```

#### C. Section-aware 查询
```python
# 根据 section 类型调整查询权重
section = query.get("section", "")
if section == "Introduction":
    # 更关注背景和动机
    query_text = f"background {citation_context} motivation"
elif section == "Related Work":
    # 更关注相关工作
    query_text = f"related work {citation_context} previous methods"
```

**预期提升**: +5-10% MRR

---

### 2. 神经伪相关反馈 (NPRF) ⭐⭐⭐⭐⭐

**参考论文**: "NPRF: A Neural Pseudo Relevance Feedback Framework for Ad-hoc Information Retrieval"

**实现思路**:
1. Stage1 检索得到 top-K 文档（K=10-20）
2. 从这些文档中提取关键术语
3. 扩展原始查询
4. 重新检索

**代码框架**:
```python
class NPRFRetriever:
    def __init__(self, base_retriever, expansion_model):
        self.base_retriever = base_retriever
        self.expansion_model = expansion_model  # 可以是 TF-IDF 或 BERT
    
    def retrieve(self, query, top_k=1000):
        # 1. 初始检索
        initial_results = self.base_retriever.retrieve(query, top_k=50)
        
        # 2. 提取扩展词
        expansion_terms = self._extract_expansion_terms(
            query, initial_results, num_terms=10
        )
        
        # 3. 构建扩展查询
        expanded_query = f"{query} {' '.join(expansion_terms)}"
        
        # 4. 重新检索
        final_results = self.base_retriever.retrieve(expanded_query, top_k=top_k)
        
        return final_results
```

**预期提升**: +8-15% MRR

---

### 3. 改进负样本采样 ⭐⭐⭐⭐

**当前问题**: 训练时只用 20 个负样本，可能不够 hard

**改进方案**:

#### A. Hard Negative Mining
```python
# 在训练过程中动态挖掘 hard negatives
def mine_hard_negatives(query, positive, all_documents, model, top_k=50):
    # 1. 用当前模型检索
    candidates = model.retrieve(query, top_k=top_k)
    
    # 2. 过滤掉 positive
    hard_negatives = [
        doc for doc, score in candidates 
        if doc['id'] != positive['id'] and score > threshold
    ]
    
    return hard_negatives[:10]  # 返回 top-10 hard negatives
```

#### B. 增加负样本数量
```python
# 当前: 20 个负样本
negatives = sample.get("negatives", [])[:20]

# 改进: 50-100 个负样本，包括不同难度
negatives = (
    sample.get("negatives", [])[:30] +  # Easy
    mine_hard_negatives(...)[:20] +      # Hard
    random_sample(...)[:20]              # Random
)
```

**预期提升**: +3-8% MRR

---

### 4. 增强特征工程 ⭐⭐⭐⭐

**当前**: 18 个特征

**新增特征**:

#### A. 文本重叠特征（更细粒度）
```python
# 1. N-gram 重叠 (bigram, trigram)
def ngram_overlap(query_text, doc_text, n=2):
    query_ngrams = set(ngrams(query_text.split(), n))
    doc_ngrams = set(ngrams(doc_text.split(), n))
    return len(query_ngrams & doc_ngrams) / len(query_ngrams | doc_ngrams)

# 2. 实体重叠（如果可用）
# 3. 关键词匹配（TF-IDF top terms）
```

#### B. 语义相似度特征（多模型）
```python
# 使用多个模型计算相似度
features = [
    specter2_sim,
    scibert_sim,
    colbert_sim,
    cross_encoder_score,
    # 新增:
    sentence_bert_sim,      # 通用 sentence transformer
    mpnet_sim,              # MPNet
    minilm_sim,             # MiniLM
]
```

#### C. 引用网络特征（如果数据可用）
```python
# 1. 共同引用数
common_citations = len(set(source_citations) & set(target_citations))

# 2. 引用路径距离
citation_distance = shortest_path(source, target)

# 3. PageRank 分数
target_pagerank = citation_graph.pagerank(target)
```

**预期提升**: +5-12% MRR

---

### 5. 优化模型融合策略 ⭐⭐⭐

**当前**: RRF 使用固定 k=60

**改进方案**:

#### A. 学习型融合（L2R 已经做了，但可以改进）
```python
# 当前 L2R 使用所有特征
# 改进: 添加特征交互项
features = [
    bm25_score,
    specter2_score,
    bm25_score * specter2_score,  # 交互特征
    bm25_score ** 2,              # 非线性
    log(specter2_score + 1),       # 变换
]
```

#### B. 加权 RRF
```python
def weighted_rrf(results_list, weights, k=60):
    """加权 RRF，不同模型权重不同"""
    scores = defaultdict(float)
    for results, weight in zip(results_list, weights):
        for rank, (doc_id, _) in enumerate(results, 1):
            scores[doc_id] += weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

#### C. 学习融合权重
```python
# 使用线性回归或神经网络学习最优权重
fusion_model = LinearRegression()
fusion_model.fit(
    X=[bm25_scores, specter2_scores, ...],
    y=ground_truth_ranks
)
```

**预期提升**: +3-7% MRR

---

### 6. 更深的 Cross-Encoder ⭐⭐⭐

**当前**: 使用 `ms-marco-MiniLM-L-12-v2` (12层)

**改进方案**:

#### A. 使用更大的模型
```python
# 当前
model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# 改进选项
model_name = "cross-encoder/ms-marco-electra-base"  # 更大
# 或
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 更快但可能略差
```

#### B. 更长的输入序列
```python
# 当前: max_length=512
# 改进: 使用更长的上下文
max_length = 768  # 或 1024（如果 GPU 内存允许）
```

**预期提升**: +2-5% MRR

---

### 7. 两阶段 Fine-tuning ⭐⭐⭐

**当前**: 直接 fine-tune 预训练模型

**改进方案**:

#### A. 领域适应 + 任务适应
```python
# Stage 1: 领域适应（在科学文献上继续预训练）
model = continue_pretraining(
    base_model="allenai/specter2",
    domain_data=large_scientific_corpus,
    epochs=1
)

# Stage 2: 任务适应（在 citation 任务上 fine-tune）
model = fine_tune(
    model=model,
    task_data=citation_data,
    epochs=3
)
```

#### B. 多任务学习
```python
# 同时学习多个相关任务
tasks = [
    "citation_recommendation",
    "paper_similarity",
    "citation_context_classification"
]
model = multi_task_learning(model, tasks)
```

**预期提升**: +3-8% MRR

---

### 8. 查询扩展和重写 ⭐⭐

**实现思路**:
```python
class QueryExpander:
    def expand(self, query):
        # 1. 同义词扩展
        synonyms = get_synonyms(query)
        
        # 2. 使用 LLM 重写（如果可用）
        rewritten = llm_rewrite(query)
        
        # 3. 实体链接
        entities = extract_entities(query)
        entity_descriptions = get_entity_descriptions(entities)
        
        return f"{query} {synonyms} {rewritten} {entity_descriptions}"
```

**预期提升**: +2-5% MRR

---

### 9. 集成学习 ⭐⭐

**实现思路**:
```python
# 训练多个不同的模型，然后集成
models = [
    train_l2r(variant="specter2_ft"),
    train_l2r(variant="scibert_ft"),
    train_l2r(variant="hybrid"),
    train_cross_encoder(),
]

# 集成预测
def ensemble_predict(query, candidates, models):
    scores = []
    for model in models:
        score = model.predict(query, candidates)
        scores.append(score)
    
    # 加权平均或投票
    final_score = weighted_average(scores, weights=[0.3, 0.3, 0.2, 0.2])
    return final_score
```

**预期提升**: +2-4% MRR

---

### 10. 数据增强 ⭐⭐

**实现思路**:
```python
# 1. 回译（如果有多语言数据）
# 2. 同义词替换
# 3. 句子重组
# 4. 负样本生成（使用生成模型生成困难负样本）
```

**预期提升**: +1-3% MRR

---

## 📈 预期累积提升

如果实施所有改进（按优先级）：

| 改进项 | 预期提升 | 累积 MRR |
|--------|---------|---------|
| 基线 | - | 0.3428 |
| 1. 查询增强 | +5-10% | 0.36-0.38 |
| 2. NPRF | +8-15% | 0.39-0.44 |
| 3. Hard Negatives | +3-8% | 0.40-0.47 |
| 4. 特征工程 | +5-12% | 0.42-0.53 |
| 5. 融合优化 | +3-7% | 0.43-0.57 |
| 6-10. 其他 | +5-10% | **0.45-0.63** |

**目标**: MRR > 0.50 (可能达到 0.55-0.60)

---

## 🎯 快速实施建议（优先级排序）

### Phase 1: 快速改进（1-2天）
1. ✅ **查询增强** - 组合 citation_context + source paper title/abstract
2. ✅ **增加负样本数量** - 从 20 增加到 50-100
3. ✅ **优化 RRF 参数** - 网格搜索 k 值

### Phase 2: 中等改进（3-5天）
4. ✅ **实现 NPRF** - 神经伪相关反馈
5. ✅ **增强特征工程** - 添加 n-gram 重叠、多模型相似度
6. ✅ **Hard Negative Mining** - 动态挖掘困难负样本

### Phase 3: 深度改进（1-2周）
7. ✅ **两阶段 Fine-tuning** - 领域适应 + 任务适应
8. ✅ **更大的 Cross-Encoder** - 使用更大的模型
9. ✅ **集成学习** - 训练多个模型并集成

---

## 📚 参考论文

1. **NPRF**: "Neural Pseudo-Relevance Feedback Models for Ad-hoc Information Retrieval" (SIGIR 2018)
2. **DeepRank**: "DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval" (CIKM 2017)
3. **GRAPHENE**: "GRAPHENE: A Context-Aware Knowledge Graph Framework for Biomedical Literature Retrieval" (arXiv 2019)
4. **COIL**: "Contextualized Inverted List for Information Retrieval" (ACL 2021)
5. **ColBERT**: "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" (SIGIR 2020)

---

## 💡 实施建议

1. **先实施 Phase 1**，预期可以快速提升到 **MRR > 0.40**
2. **然后实施 Phase 2**，预期可以达到 **MRR > 0.45-0.50**
3. **最后实施 Phase 3**，目标 **MRR > 0.55**

每一步都要做 A/B 测试，确保改进有效！

