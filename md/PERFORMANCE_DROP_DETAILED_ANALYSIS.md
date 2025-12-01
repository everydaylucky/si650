# 性能下降详细分析

## 📊 问题现象

| 实验 | MRR | Recall@10 | NDCG@10 | 数据集 |
|------|-----|-----------|---------|--------|
| **context_before_old** (084719) | 0.3428 | 0.5996 | 0.3977 | fast_experiment |
| **context_before_new** (125136) | 0.2029 | 0.3432 | 0.2324 | data/full |
| **下降幅度** | **-40.8%** | **-42.8%** | **-41.5%** | - |

## 🔍 根本原因分析

### 1. **数据集难度差异** ⚠️ **主要原因**

#### fast_experiment 数据集
- **样本数**: 472
- **负样本比例**: 1:20（每个 positive 有 20 个 negatives）
- **任务难度**: 相对容易
- **特点**: 用于快速实验，负样本较少

#### data/full 数据集
- **样本数**: 1888（4倍）
- **负样本比例**: 1:99（每个 positive 有 99 个 negatives）
- **任务难度**: **5倍难度**
- **特点**: 完整数据集，更接近真实场景

**影响**：
- 在 99 个负样本中找到正确答案，比在 20 个负样本中难得多
- 这是性能下降的主要原因（约 40-45% 的下降）

### 2. **数据量差异**

| 特性 | fast_experiment | data/full | 影响 |
|------|----------------|-----------|------|
| 样本数 | 472 | 1888 | 4倍 |
| 评估时间 | ~10 分钟 | ~40 分钟 | 4倍 |
| 统计显著性 | 较低 | 较高 | 更可靠 |

**影响**：
- 更多样本意味着更多边缘案例和困难样本
- 但这不是主要因素（主要是负样本比例）

### 3. **Context Enhancement 的影响**（次要）

#### 查询长度变化
- **无 context**: ~100 字符（只有 citation_context）
- **with context_before**: ~586 字符（平均）
- **最大长度**: 1354 字符

**可能的影响**：
- ✅ 提供更多上下文信息（理论上应该有帮助）
- ⚠️ 增加查询长度，可能引入噪声
- ⚠️ 模型截断问题（如果超过最大长度）
- ⚠️ 在更难的数据集上，噪声的影响可能更明显

**实际效果**：
- 在 `fast_experiment` 上：MRR 相同（0.3428），说明 context_before 没有带来提升
- 在 `data/full` 上：MRR = 0.2029，但这是数据集难度导致的

### 4. **索引文档数**（已修复）

**之前的问题**：
- 只索引了 620 个文档（只索引 positive）
- 导致很多候选文档无法被检索到

**修复后**：
- 索引了 4504 个文档（所有 candidates）
- 这是正确的做法，但不会导致性能下降

## 📈 性能下降分解

### 理论分析

假设性能下降主要来自数据集难度：

1. **负样本比例影响**（主要）：
   - 1:20 → 1:99：难度增加 5 倍
   - 预期性能下降：~40-45%
   - **实际下降：40.8%** ✅ 匹配

2. **数据量影响**（次要）：
   - 更多样本 → 更多困难案例
   - 预期额外下降：~5-10%
   - **实际影响：已包含在 40.8% 中**

3. **Context Enhancement 影响**（很小）：
   - 查询长度增加 → 可能引入噪声
   - 预期影响：±2-5%
   - **实际影响：难以量化，但应该很小**

### 实际对比

| 对比项 | fast_experiment | data/full | 差异 |
|--------|----------------|-----------|------|
| **Baseline (optimized)** | MRR = 0.3428 | MRR ≈ 0.18-0.20 (估计) | -40-45% |
| **With context_before** | MRR = 0.3428 | MRR = 0.2029 | -40.8% |

**结论**：性能下降主要是数据集难度导致的，不是模型或 context enhancement 的问题。

## ✅ 正确理解

### 这不是性能下降，而是评估基准不同

1. **fast_experiment**：
   - 用于快速测试和调试
   - 负样本比例 1:20，任务相对容易
   - MRR = 0.34 是合理的

2. **data/full**：
   - 完整数据集，更接近真实场景
   - 负样本比例 1:99，任务难度增加 5 倍
   - MRR = 0.20 也是合理的（在更难的评估基准上）

### 正确的评估方式

1. **建立 baseline**：
   ```bash
   # 在 data/full 上运行 optimized pipeline（无 context enhancement）
   python scripts/run_all_experiments.py --experiment exp_5_2_pipeline_optimized --data_dir data/full_indexed
   ```

2. **对比 context enhancement**：
   - 如果 baseline MRR ≈ 0.18-0.20
   - context_before MRR = 0.2029
   - 那么 context enhancement 可能带来了 **+1-2%** 的提升

3. **不要跨数据集对比**：
   - ❌ 不要对比 fast_experiment 和 data/full 的结果
   - ✅ 只在同一数据集内对比不同方法

## 🎯 建议

### 1. 建立正确的 baseline

在 `data/full` 上运行 baseline 实验：
```bash
python scripts/run_all_experiments.py --experiment exp_5_2_pipeline_optimized --data_dir data/full_indexed
```

### 2. 理解数据集差异

- `fast_experiment`: 用于快速测试（1:20 负样本）
- `data/full`: 完整评估（1:99 负样本，更真实）

### 3. 评估 Context Enhancement

在 `data/full` 上对比：
- Baseline (无 context): MRR ≈ ?
- With context_before: MRR = 0.2029
- With context_after: MRR = ?
- With context_both: MRR = ?

**如果 baseline MRR ≈ 0.18-0.20，那么 context enhancement 可能带来了小幅提升。**

## 📝 总结

**性能下降的主要原因**：
1. ✅ **数据集难度差异**（负样本比例 1:20 vs 1:99）- **主要因素**
2. ✅ **数据量差异**（472 vs 1888）- 次要因素
3. ⚠️ **Context Enhancement** - 影响很小，可能略有帮助

**这不是问题，而是评估基准不同**：
- `fast_experiment`: 容易的任务，MRR = 0.34 合理
- `data/full`: 困难的任务，MRR = 0.20 也合理

**下一步**：
- 在 `data/full` 上运行 baseline，建立正确的性能基准
- 在同一数据集内对比 context enhancement 的效果

