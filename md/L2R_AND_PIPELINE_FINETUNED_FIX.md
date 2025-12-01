# L2R 和 Multi-Stage Pipeline Fine-tuned 修复

## 🔍 问题诊断

### 1. L2R Fine-tuned (exp_4_4_l2r_ft)
**问题**: 
- 虽然标记为 "fine-tuned"，但训练时只使用了 zero-shot SPECTER2
- 没有使用 SciBERT Fine-tuned 或 SPECTER2 Fine-tuned 的 embedding
- 配置文件中的 `fine_tuned_path` 都是 `null`

### 2. Multi-Stage Pipeline Optimized (exp_5_2_pipeline_optimized)
**问题**:
- 虽然描述说 "with fine-tuned models"，但配置中所有 `fine_tuned_path` 都是 `null`
- 实际上没有使用任何 fine-tuned 模型

## ✅ 修复内容

### 1. 修复 L2R 训练脚本 (`scripts/train_l2r.py`)

**改进**:
- 在 `extract_features_for_training` 函数中，当 `use_fine_tuned=True` 时：
  - 使用 SPECTER2 Fine-tuned 模型（如果存在）
  - 使用 SciBERT Fine-tuned 模型（如果存在）
  - 计算真实的 embedding 相似度特征，而不是占位符

**代码变更**:
```python
if use_fine_tuned:
    # 使用Fine-tuned SPECTER2
    specter2_ft_path = project_root / "experiments" / "checkpoints" / "specter2"
    if specter2_ft_path.exists():
        dense = DenseRetriever(
            model_name="allenai/specter2_base",
            fine_tuned_path=str(specter2_ft_path)
        )
        dense.build_index(unique_docs)
    
    # 使用Fine-tuned SciBERT
    scibert_ft_path = project_root / "experiments" / "checkpoints" / "scibert"
    if scibert_ft_path.exists():
        scibert_ft = BiEncoder(
            model_name="allenai/scibert_scivocab_uncased",
            fine_tuned_path=str(scibert_ft_path)
        )
    
    # 在特征提取时计算真实的embedding相似度
    if dense:
        query["_specter2_score"] = ...  # 真实分数
    if scibert_ft:
        query["_scibert_score"] = ...  # 真实分数
```

### 2. 更新 L2R Fine-tuned 配置 (`config/experiments/exp_4_4_l2r_ft.yaml`)

**改进**:
- 设置 `specter2.fine_tuned_path: experiments/checkpoints/specter2`
- 设置 `bi_encoder.fine_tuned_path: experiments/checkpoints/scibert`
- 设置 `cross_encoder.fine_tuned_path: experiments/checkpoints/cross_encoder`
- 启用必要的 Stage1 检索器（BM25, SPECTER2, TF-IDF）以提供候选
- 启用 Stage2 的 SciBERT 以提供 embedding 特征
- 设置 L2R 模型路径

### 3. 更新 Multi-Stage Pipeline Optimized 配置 (`config/experiments/exp_5_2_pipeline_optimized.yaml`)

**改进**:
- 设置所有 fine-tuned 模型路径：
  - `specter2.fine_tuned_path: experiments/checkpoints/specter2`
  - `bi_encoder.fine_tuned_path: experiments/checkpoints/scibert`
  - `cross_encoder.fine_tuned_path: experiments/checkpoints/cross_encoder`
  - `l2r.model_path: experiments/checkpoints/l2r/ft/l2r_model.txt`
- 启用所有阶段：
  - Stage1: BM25 + SPECTER2 Fine-tuned + TF-IDF
  - Stage2: SciBERT Fine-tuned + RRF
  - Stage3: Cross-Encoder Fine-tuned + L2R Fine-tuned

## 🎯 预期效果

### L2R Fine-tuned
- **训练时**: 使用 fine-tuned 模型的真实 embedding 特征
- **评估时**: Pipeline 使用 fine-tuned 模型提供 embedding 分数
- **预期性能**: MRR 应该比 zero-shot L2R 提升 5-15%

### Multi-Stage Pipeline Optimized
- **所有阶段**: 使用 fine-tuned 模型
- **预期性能**: 应该是最佳性能，MRR 可能达到 0.45-0.50

## 📝 下一步

1. **重新训练 L2R Fine-tuned**:
   ```bash
   python scripts/run_all_experiments.py --experiment exp_4_4_l2r_ft --data_dir data/processed/fast_experiment
   ```

2. **运行 Multi-Stage Pipeline Optimized**:
   ```bash
   python scripts/run_all_experiments.py --experiment exp_5_2_pipeline_optimized --data_dir data/processed/fast_experiment
   ```

3. **对比分析**:
   - L2R Zero-shot vs L2R Fine-tuned
   - Pipeline Basic vs Pipeline Optimized
   - 所有实验的最终排名

## ⚠️ 注意事项

1. **模型依赖**: 
   - L2R Fine-tuned 需要先训练好 SPECTER2 Fine-tuned 和 SciBERT Fine-tuned
   - Pipeline Optimized 需要所有 fine-tuned 模型都已训练完成

2. **训练时间**:
   - L2R Fine-tuned 训练时间会增加（需要计算 embedding）
   - 预计 2-3 小时（取决于数据量和模型加载时间）

3. **特征一致性**:
   - 训练时和评估时的特征提取逻辑需要一致
   - 确保 Pipeline 配置正确传递 embedding 分数
