# SPECTER2 Fine-tuned 和 L2R Embedding 特征分析

## 🔍 SPECTER2 Fine-tuned 结果分析

### 当前问题
SPECTER2 Fine-tuned 的实验结果显示所有指标都是 0.0，这表明评估时可能没有正确加载 fine-tuned 模型。

### 可能的原因
1. **模型路径问题**: 评估时可能没有正确传递 `fine_tuned_path` 给 `DenseRetriever`
2. **模型加载失败**: Fine-tuned 模型可能没有正确保存或加载
3. **索引构建问题**: 使用 fine-tuned 模型构建索引时可能出错

### 需要检查
```bash
# 检查模型文件是否存在
ls -lh experiments/checkpoints/specter2/

# 检查配置文件中的路径
grep -r "fine_tuned_path" config/experiments/exp_3_2_specter2_ft.yaml
```

---

## 📊 L2R Embedding 特征使用情况

### 当前实现

#### 1. L2R 训练时的 Embedding 特征

**Zero-shot L2R** (`exp_4_3_l2r_zs`):
- ❌ **不使用任何 embedding 特征**
- 代码中 `use_fine_tuned=False` 时，跳过 DenseRetriever 初始化
- Embedding 特征都是占位符（0.0）

**Fine-tuned L2R** (`exp_4_4_l2r_ft`):
- ⚠️ **只使用 zero-shot SPECTER2 的 embedding**
- 代码中虽然 `use_fine_tuned=True`，但初始化的是：
  ```python
  dense = DenseRetriever()  # 默认使用 zero-shot SPECTER2
  ```
- **没有使用 fine-tuned 模型的 embedding！**

#### 2. L2R 评估时的 Embedding 特征

在 `MultiStagePipeline` 中，L2R 的 embedding 特征来自：
- `query.get("_specter2_score", 0.0)` - 来自 Stage1 的 SPECTER2 检索
- `query.get("_scibert_score", 0.0)` - 来自 Stage2 的 SciBERT 重排序
- `query.get("_colbert_score", 0.0)` - 来自 Stage2 的 ColBERT 重排序
- `query.get("_cross_encoder_score", 0.0)` - 来自 Stage3 的 Cross-Encoder 排序

**问题**: 这些分数来自 pipeline 中的各个阶段，如果某个阶段使用了 fine-tuned 模型，分数会反映 fine-tuned 的效果。但如果某个阶段没有启用，对应的 embedding 特征就是 0.0。

---

## 🎯 改进建议

### 1. 修复 SPECTER2 Fine-tuned 评估问题

**问题**: 评估时没有使用 fine-tuned 模型

**解决方案**:
1. 检查 `exp_3_2_specter2_ft.yaml` 配置中是否设置了 `fine_tuned_path`
2. 确保 `DenseRetriever` 在评估时正确加载 fine-tuned 模型
3. 验证模型文件路径是否正确

### 2. 改进 L2R 的 Embedding 特征提取

#### 方案A: 在 L2R 训练时使用 Fine-tuned 模型的 Embedding

**当前问题**:
- Fine-tuned L2R 训练时只使用了 zero-shot SPECTER2
- 没有使用 SciBERT Fine-tuned、SPECTER2 Fine-tuned、Cross-Encoder Fine-tuned 的 embedding

**改进方案**:
```python
# 在 train_l2r.py 的 extract_features_for_training 中
if use_fine_tuned:
    # 使用 fine-tuned SPECTER2
    specter2_ft_path = "experiments/checkpoints/specter2"
    dense = DenseRetriever(
        model_name="allenai/specter2_base",
        fine_tuned_path=specter2_ft_path
    )
    
    # 使用 fine-tuned SciBERT（如果需要）
    from src.models.reranking.bi_encoder import BiEncoder
    scibert_ft = BiEncoder(
        model_name="allenai/scibert_scivocab_uncased",
        fine_tuned_path="experiments/checkpoints/scibert"
    )
    
    # 计算 embedding 相似度特征
    # 然后更新 query["_specter2_score"] 和 query["_scibert_score"]
```

#### 方案B: 在 Pipeline 中传递 Fine-tuned 模型的分数

**当前实现**: Pipeline 已经支持 fine-tuned 模型，如果配置正确，分数会自动传递。

**需要确保**:
- Pipeline 配置中启用了 fine-tuned 模型
- 各阶段的 fine-tuned 模型路径正确
- L2R 的 feature extractor 能正确获取这些分数

---

## 🔬 建议的新实验

### 1. L2R with Fine-tuned Embeddings ⭐⭐⭐⭐⭐

**实验名称**: LightGBM L2R (Fine-tuned Embeddings)

**描述**: 使用 fine-tuned 模型的 embedding 特征训练 L2R

**需要修改**:
- `scripts/train_l2r.py`: 在训练时使用 fine-tuned 模型计算 embedding 特征
- 需要加载：
  - SPECTER2 Fine-tuned
  - SciBERT Fine-tuned
  - Cross-Encoder Fine-tuned（可选）

**预期效果**: 应该比当前的 L2R Fine-tuned 性能更好

### 2. Ablation Study: Embedding Features ⭐⭐⭐⭐

**实验**: 消融研究 - 移除 embedding 特征

**方法**: 
- 训练一个只使用 IR + Category + Temporal + Context 特征的 L2R 模型
- 对比完整特征集 vs 无 embedding 特征集

**目的**: 评估 embedding 特征的贡献

### 3. Ablation Study: Individual Embedding Features ⭐⭐⭐

**实验**: 消融研究 - 逐个移除 embedding 特征

**方法**:
- 移除 SPECTER2 特征
- 移除 SciBERT 特征
- 移除 ColBERT 特征
- 移除 Cross-Encoder 特征

**目的**: 找出最重要的 embedding 特征

### 4. Feature Importance Analysis ⭐⭐⭐⭐

**实验**: 特征重要性分析

**方法**:
- 使用 LightGBM 的 `feature_importance()` 方法
- 分析18个特征的重要性排名
- 可视化特征重要性

**目的**: 理解哪些特征对排序最重要

---

## 📋 实施优先级

### 优先级1: 修复 SPECTER2 Fine-tuned ⭐⭐⭐⭐⭐
1. 检查模型文件是否存在
2. 检查配置文件中的路径
3. 修复评估时的模型加载逻辑
4. 重新运行实验

### 优先级2: 改进 L2R Fine-tuned Embeddings ⭐⭐⭐⭐
1. 修改 `train_l2r.py` 使用 fine-tuned 模型
2. 重新训练 L2R Fine-tuned
3. 对比性能改进

### 优先级3: 特征重要性分析 ⭐⭐⭐
1. 提取 LightGBM 特征重要性
2. 生成可视化图表
3. 分析结果

### 优先级4: 消融实验 ⭐⭐
1. 设计消融实验
2. 运行实验
3. 分析结果

---

## 🚀 快速修复命令

### 检查 SPECTER2 Fine-tuned 模型
```bash
cd /hy-tmp/final_test

# 检查模型文件
ls -lh experiments/checkpoints/specter2/

# 检查配置
cat config/experiments/exp_3_2_specter2_ft.yaml | grep -A 5 "specter2"
```

### 重新运行 SPECTER2 Fine-tuned 评估
```bash
# 如果模型文件存在，可以直接重新评估
python scripts/run_all_experiments.py \
    --experiment exp_3_2_specter2_ft \
    --data_dir data/processed/fast_experiment
```

---

## 📊 当前 L2R 特征总结

### 18个特征分布

1. **IR特征 (4个)**: BM25, TF-IDF, Title overlap, Abstract overlap
2. **Embedding特征 (4个)**: 
   - SPECTER2 similarity (zero-shot 或 fine-tuned，取决于 pipeline 配置)
   - SciBERT similarity (zero-shot 或 fine-tuned，取决于 pipeline 配置)
   - ColBERT score (zero-shot)
   - Cross-Encoder score (zero-shot 或 fine-tuned，取决于 pipeline 配置)
3. **Category特征 (4个)**: Primary match, Overlap ratio, Same AI/ML domain, Related match
4. **Temporal特征 (3个)**: Year difference, Recency, Is recent
5. **Context特征 (3个)**: Citation length, Abstract ratio, Num categories

### 问题总结

1. **训练时**: Fine-tuned L2R 没有使用 fine-tuned 模型的 embedding
2. **评估时**: Embedding 特征来自 pipeline 各阶段，如果阶段未启用，特征为 0.0
3. **改进空间**: 可以在训练时显式使用 fine-tuned 模型计算 embedding 特征

