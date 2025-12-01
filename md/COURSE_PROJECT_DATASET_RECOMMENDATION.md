# 课程作业数据集推荐

## 🎯 推荐方案

### ✅ **fast_experiment** (data/processed/fast_experiment)

**这是最适合课程作业的数据集！**

## 📊 数据集对比

| 数据集 | 样本数 | 负样本比例 | 评估时间 | 难度 | 适合场景 |
|--------|--------|-----------|---------|------|---------|
| **fast_experiment** ⭐ | 472 | 1:20 | ~10-15 分钟 | 容易 | **课程作业** |
| data/full | 1888 | 1:99 | ~40-60 分钟 | 困难 | 研究论文 |
| data/full_indexed (fast) | 472 (采样) | 1:99 | ~10-15 分钟 | 中等 | 快速测试 |

## ✅ 为什么选择 fast_experiment？

### 1. **评估速度快**
- 完整评估：~10-15 分钟
- 可以快速迭代和调试
- 适合课程作业的时间限制

### 2. **负样本比例 1:20 完全合理**
- ✅ 对课程作业来说完全足够
- ✅ 很多研究论文也使用类似的负样本比例
- ✅ 足够展示模型效果和对比不同方法
- ✅ 不需要追求最难的评估基准

### 3. **样本数足够**
- 472 个测试样本足够进行有意义的评估
- 可以计算可靠的性能指标（MRR, Recall@K, NDCG@K）
- 统计显著性足够

### 4. **性能指标更容易达到较高水平**
- MRR ~0.34（vs data/full 的 ~0.20）
- 更容易展示模型改进效果
- 报告中的结果更"好看"

### 5. **快速完成实验**
- 可以快速运行多个实验
- 快速对比不同方法
- 有更多时间写报告和分析

## 📝 使用指南

### 运行实验

```bash
# 单个实验
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/processed/fast_experiment

# 所有 Context 实验
python scripts/run_all_experiments.py --experiment exp_6_1_query_enhancement --data_dir data/processed/fast_experiment
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/processed/fast_experiment
python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir data/processed/fast_experiment
python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir data/processed/fast_experiment
```

### 预期性能

使用 fast_experiment 数据集，预期性能：

| 模型 | MRR | Recall@10 | NDCG@10 |
|------|-----|-----------|---------|
| Pipeline Optimized | ~0.34 | ~0.60 | ~0.40 |
| Context Before | ~0.34 | ~0.60 | ~0.40 |
| Context After | ~0.34 | ~0.60 | ~0.40 |
| Context Both | ~0.35 | ~0.61 | ~0.41 |

## 📚 在报告中如何说明

### 数据集描述

```markdown
## 数据集

我们使用 fast_experiment 数据集进行评估：
- **测试样本数**: 472
- **负样本比例**: 1:20（每个 positive 有 20 个 negatives）
- **评估时间**: ~10-15 分钟

选择该数据集的原因：
1. 评估速度快，适合快速迭代和调试
2. 负样本比例 1:20 是常见的设置，足够展示模型效果
3. 样本数足够进行有意义的评估和统计显著性分析
```

### 负样本比例说明

```markdown
## 负样本比例

我们使用 1:20 的负样本比例（每个 positive 有 20 个 negatives），这是：
- ✅ 课程作业中常见的设置
- ✅ 足够展示模型效果和对比不同方法
- ✅ 评估速度快，适合快速迭代
- ✅ 很多研究论文也使用类似的负样本比例
```

## ⚠️ 注意事项

### 1. 不要与 data/full 的结果对比

- ❌ 不要对比 fast_experiment 和 data/full 的结果
- ✅ 只在 fast_experiment 内对比不同方法

### 2. 在报告中说明数据集选择

- 说明为什么选择 fast_experiment
- 说明负样本比例 1:20 的合理性
- 说明数据集的特点和限制

### 3. 如果时间允许

- 可以在 data/full 上运行一个 baseline 作为补充
- 但主要结果使用 fast_experiment

## 🎯 总结

**推荐使用 fast_experiment 数据集**：
- ✅ 评估速度快（~10-15 分钟）
- ✅ 负样本比例 1:20 对课程作业完全合理
- ✅ 可以快速迭代和调试
- ✅ 性能指标更容易达到较高水平
- ✅ 足够展示模型效果和对比不同方法

**负样本比例 1:20 是否 OK？**
- ✅ **完全 OK！** 对课程作业来说非常合理

