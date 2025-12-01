# SPECTER2 Fine-tuned 问题修复总结

## 🔍 问题诊断

### 症状
- SPECTER2 Fine-tuned 实验结果：**MRR = 0.0**
- 所有指标都是 0.0
- 评估速度异常快（326308.14it/s），说明没有真正进行检索

### 根本原因

1. **配置文件问题**: `config/experiments/exp_3_2_specter2_ft.yaml` 中 `use_specter2: false`
2. **评估时未自动修复**: 虽然训练时模型已保存，但评估时配置中 `use_specter2: false`，导致：
   - `MultiStagePipeline` 没有初始化 `DenseRetriever`
   - `stage1_retrievers` 为空列表
   - `pipeline.retrieve()` 返回空列表（因为没有检索器）
   - 所有 predictions 都是空列表
   - MRR = 0.0

### 代码流程分析

```
评估流程:
1. 加载配置文件 → use_specter2: false
2. 初始化 MultiStagePipeline → stage1_retrievers = [] (空)
3. 构建索引 → 跳过（因为没有检索器需要索引）
4. pipeline.retrieve(query) → stage1_results = [] → candidate_docs = [] → 返回 []
5. 所有 predictions = [] → MRR = 0.0
```

---

## ✅ 修复方案

### 修复1: 配置文件
- ✅ 已修复 `config/experiments/exp_3_2_specter2_ft.yaml`
- 将 `use_specter2: false` 改为 `use_specter2: true`

### 修复2: 自动修复逻辑
- ✅ 在 `scripts/run_all_experiments.py` 的 `run_single_experiment` 中添加：
  ```python
  # 对于specter2实验，必须启用use_specter2
  if model_type == "specter2":
      if not stage1_config.get("use_specter2", False):
          print(f"⚠ 检测到SPECTER2实验，自动启用use_specter2...")
          stage1_config["use_specter2"] = True
  ```

这样即使配置文件中有问题，代码也会自动修复。

---

## 🚀 重新运行

现在可以重新运行实验：

```bash
cd /hy-tmp/final_test
python scripts/run_all_experiments.py --experiment exp_3_2_specter2_ft --data_dir data/processed/fast_experiment
```

**预期结果**:
- 应该能看到 "⚠ 检测到SPECTER2实验，自动启用use_specter2..." 的提示
- 评估时会真正使用 SPECTER2 Fine-tuned 模型进行检索
- MRR 应该 > 0.0（预期在 0.28-0.35 之间，比 zero-shot 的 0.28 略高）

---

## 📊 关于 L2R Embedding 特征

### 当前实现

**Zero-shot L2R**:
- ❌ 不使用 embedding 特征（都是 0.0）
- 只使用 IR + Category + Temporal + Context 特征

**Fine-tuned L2R**:
- ⚠️ **只使用 zero-shot SPECTER2 的 embedding**
- 代码中虽然 `use_fine_tuned=True`，但初始化的是：
  ```python
  dense = DenseRetriever()  # 默认 zero-shot，没有 fine_tuned_path
  ```
- **没有使用 fine-tuned 模型的 embedding！**

### 改进建议

可以在 `train_l2r.py` 中改进，让 Fine-tuned L2R 使用 fine-tuned 模型的 embedding：

```python
if use_fine_tuned:
    # 使用 fine-tuned SPECTER2
    specter2_ft_path = project_root / "experiments" / "checkpoints" / "specter2"
    dense = DenseRetriever(
        model_name="allenai/specter2_base",
        fine_tuned_path=str(specter2_ft_path)
    )
    
    # 使用 fine-tuned SciBERT
    from src.models.reranking.bi_encoder import BiEncoder
    scibert_ft = BiEncoder(
        model_name="allenai/scibert_scivocab_uncased",
        fine_tuned_path=str(project_root / "experiments" / "checkpoints" / "scibert")
    )
    
    # 计算 embedding 相似度并更新 query 中的分数
    # query["_specter2_score"] = ...
    # query["_scibert_score"] = ...
```

---

## 🎯 其他可以做的实验

### 1. 修复并重新运行 SPECTER2 Fine-tuned ⭐⭐⭐⭐⭐
```bash
python scripts/run_all_experiments.py --experiment exp_3_2_specter2_ft --data_dir data/processed/fast_experiment
```

### 2. L2R with Fine-tuned Embeddings ⭐⭐⭐⭐
改进 `train_l2r.py`，使用 fine-tuned 模型的 embedding 特征

### 3. 特征重要性分析 ⭐⭐⭐
分析 LightGBM L2R 中 18 个特征的重要性

### 4. 消融实验 ⭐⭐
- 移除 embedding 特征，看性能下降
- 逐个移除 embedding 特征，找出最重要的

### 5. 生成最终分析报告 ⭐⭐⭐⭐⭐
```bash
python scripts/analyze_results.py
```

---

## ✅ 修复检查清单

- [x] 修复配置文件中的 `use_specter2: false` → `true`
- [x] 添加自动修复逻辑（如果 model_type == "specter2"，自动启用 use_specter2）
- [ ] 重新运行实验验证修复
- [ ] 检查结果是否正常（MRR > 0）

