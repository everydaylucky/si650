# 统一 Fast 模式实验运行指南

## 🎯 设计原则

**确保所有实验在完全相同的条件下运行，保证对比的公平性！**

## 📊 推荐方案

### ✅ **使用 fast_experiment 数据集**

**原因**：
1. 数据集本身就是为快速实验设计的（472 个样本，负样本比例 1:20）
2. 所有实验条件完全相同，对比最公平
3. 不需要采样，避免随机性
4. 评估速度快（~10-15 分钟）
5. 负样本比例 1:20 对课程作业合理

## 🚀 运行命令

### 方案1: 使用 fast_experiment 数据集（推荐）⭐

```bash
# Baseline: Optimized Pipeline
python scripts/run_all_experiments.py --experiment exp_5_2_pipeline_optimized --data_dir data/processed/fast_experiment

# Context Enhancement 实验
python scripts/run_all_experiments.py --experiment exp_6_1_query_enhancement --data_dir data/processed/fast_experiment
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/processed/fast_experiment
python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir data/processed/fast_experiment
python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir data/processed/fast_experiment
```

### 方案2: 使用 data/full_indexed + --fast（备选）

如果必须使用 data/full 数据集，可以使用采样模式：

```bash
# Baseline: Optimized Pipeline
python scripts/run_all_experiments.py --experiment exp_5_2_pipeline_optimized --data_dir data/full_indexed --fast

# Context Enhancement 实验
python scripts/run_all_experiments.py --experiment exp_6_1_query_enhancement --data_dir data/full_indexed --fast
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/full_indexed --fast
python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir data/full_indexed --fast
python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir data/full_indexed --fast
```

**注意**：使用 `--fast` 时，所有实验会使用相同的随机种子（42），确保采样一致。

## 📝 批量运行脚本

### 使用 fast_experiment（推荐）

```bash
#!/bin/bash
# run_all_fast_experiments.sh

DATA_DIR="data/processed/fast_experiment"

echo "开始运行所有 Fast 模式实验..."
echo "数据集: $DATA_DIR"
echo ""

# Baseline
echo "运行 Baseline: Optimized Pipeline"
python scripts/run_all_experiments.py --experiment exp_5_2_pipeline_optimized --data_dir $DATA_DIR

# Context Enhancement 实验
echo ""
echo "运行 Context Enhancement 实验..."
python scripts/run_all_experiments.py --experiment exp_6_1_query_enhancement --data_dir $DATA_DIR
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir $DATA_DIR
python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir $DATA_DIR
python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir $DATA_DIR

echo ""
echo "所有实验完成！"
```

### 使用 data/full_indexed + --fast（备选）

```bash
#!/bin/bash
# run_all_fast_experiments_sampled.sh

DATA_DIR="data/full_indexed"
RANDOM_SEED=42

echo "开始运行所有 Fast 模式实验（采样模式）..."
echo "数据集: $DATA_DIR"
echo "采样: 472 个样本"
echo "随机种子: $RANDOM_SEED"
echo ""

# Baseline
echo "运行 Baseline: Optimized Pipeline"
python scripts/run_all_experiments.py --experiment exp_5_2_pipeline_optimized --data_dir $DATA_DIR --fast --random_seed $RANDOM_SEED

# Context Enhancement 实验
echo ""
echo "运行 Context Enhancement 实验..."
python scripts/run_all_experiments.py --experiment exp_6_1_query_enhancement --data_dir $DATA_DIR --fast --random_seed $RANDOM_SEED
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir $DATA_DIR --fast --random_seed $RANDOM_SEED
python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir $DATA_DIR --fast --random_seed $RANDOM_SEED
python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir $DATA_DIR --fast --random_seed $RANDOM_SEED

echo ""
echo "所有实验完成！"
```

## 🔍 验证实验条件一致性

运行后，检查所有实验结果，确保：

1. **数据集相同**：
   - 如果使用 fast_experiment：所有实验的 `data_dir` 都是 `data/processed/fast_experiment`
   - 如果使用 data/full_indexed + --fast：所有实验都有 `fast_mode` 字段

2. **样本数相同**：
   - fast_experiment：所有实验都是 472 个样本
   - data/full_indexed + --fast：所有实验的 `fast_mode.sample_size` 都是 472

3. **随机种子相同**（如果使用采样）：
   - 所有实验的 `fast_mode.random_seed` 都是 42

## 📊 预期结果对比

使用 fast_experiment 数据集，预期性能：

| 实验 | MRR | Recall@10 | NDCG@10 | 说明 |
|------|-----|-----------|---------|------|
| Pipeline Optimized | ~0.34 | ~0.60 | ~0.40 | Baseline |
| Query Enhancement | ~0.34-0.35 | ~0.60-0.61 | ~0.40-0.41 | +source_paper |
| Context Before | ~0.34-0.35 | ~0.60-0.61 | ~0.40-0.41 | +context_before |
| Context After | ~0.34-0.35 | ~0.60-0.61 | ~0.40-0.41 | +context_after |
| Context Both | ~0.35-0.36 | ~0.61-0.62 | ~0.41-0.42 | +context_both |

## ⚠️ 重要注意事项

### 1. 确保所有实验使用相同数据集

```bash
# ✅ 正确：所有实验都用 fast_experiment
--data_dir data/processed/fast_experiment

# ❌ 错误：混用不同数据集
--data_dir data/processed/fast_experiment  # 实验1
--data_dir data/full_indexed --fast        # 实验2（不能对比！）
```

### 2. 如果使用采样模式，确保随机种子相同

```bash
# ✅ 正确：所有实验都用相同的随机种子
--fast --random_seed 42

# ❌ 错误：使用不同的随机种子
--fast --random_seed 42  # 实验1
--fast --random_seed 123 # 实验2（采样不同，不能对比！）
```

### 3. 在报告中说明

```markdown
## 实验设置

所有实验在相同的条件下运行：
- **数据集**: fast_experiment (472 个测试样本)
- **负样本比例**: 1:20
- **评估时间**: ~10-15 分钟/实验

这确保了所有实验的对比是公平的。
```

## 🎯 推荐执行步骤

1. **选择方案1**（fast_experiment 数据集）
2. **运行所有实验**（使用上面的批量脚本）
3. **检查结果一致性**（确保所有实验条件相同）
4. **对比分析**（在同一数据集内对比不同方法）

## 📈 结果分析

运行完所有实验后，可以：

1. **对比不同 Context Enhancement 方法**：
   - Context Before vs Context After vs Context Both
   - 看哪个方法效果最好

2. **对比 Baseline vs Enhancement**：
   - Optimized Pipeline vs Context Enhancement
   - 看 Context Enhancement 是否带来提升

3. **统计分析**：
   - 使用 `scripts/analyze_results.py` 分析所有结果
   - 生成对比表格和图表

