# Fast 模式设计方案

## 📊 需求分析

### 当前情况
- **data/full**: 1888 个测试样本，完整评估需要较长时间
- **fast_experiment**: 472 个样本（25%），用于快速测试
- **评估时间**: 完整评估可能需要 30-60 分钟，fast 模式约 8-15 分钟

### 目标
- 提供快速测试模式，减少评估时间
- 保持结果的可重复性
- 不影响正常实验流程

## 🎯 设计方案

### 方案1：采样测试样本 + 完整索引（推荐）⭐

**设计**：
- 从 `test.json` 中随机采样 N 个样本（默认 472 个）
- 保持完整的 `corpus.json`（4504 个文档）
- 使用固定随机种子确保可重复性

**优点**：
- ✅ 实现简单，只需采样测试数据
- ✅ 索引完整，ground truth 肯定在索引中
- ✅ 不改变数据文件，只影响评估
- ✅ 可以灵活调整采样数量

**缺点**：
- ⚠️ 索引包含所有文档（但这不是问题，只是稍慢）

**实现位置**：
- 在 `evaluate_experiment` 函数中，加载测试数据后采样

### 方案2：创建 Fast 版本的完整数据集

**设计**：
- 采样测试样本（472 个）
- 只索引这些样本相关的文档（positive + negatives）
- 创建新的 `corpus_fast.json` 和 `test_fast.json`

**优点**：
- ✅ 更接近真实 fast_experiment
- ✅ 索引更小，构建更快

**缺点**：
- ❌ 需要重新组织数据
- ❌ 需要维护两套数据文件
- ❌ 更复杂

### 方案3：混合方案

**设计**：
- 采样测试样本
- 索引所有文档（保持完整）
- 在结果中标记使用了 fast 模式

**优点**：
- ✅ 简单且完整
- ✅ 结果可追溯

**缺点**：
- ⚠️ 与方案1类似，但增加了标记

## 🚀 推荐实现：方案1

### 参数设计

```bash
# 方式1: 使用 --fast 标志（默认采样 472 个）
python scripts/run_all_experiments.py \
    --experiment exp_6_1b_1_context_before \
    --data_dir data/full_indexed \
    --fast

# 方式2: 指定采样数量
python scripts/run_all_experiments.py \
    --experiment exp_6_1b_1_context_before \
    --data_dir data/full_indexed \
    --sample_size 500

# 方式3: 指定采样比例
python scripts/run_all_experiments.py \
    --experiment exp_6_1b_1_context_before \
    --data_dir data/full_indexed \
    --sample_ratio 0.25  # 25%
```

### 实现细节

1. **采样逻辑**：
   ```python
   import random
   
   def sample_test_data(test_data, sample_size=None, sample_ratio=None, random_seed=42):
       random.seed(random_seed)
       
       if sample_size:
           n = min(sample_size, len(test_data))
       elif sample_ratio:
           n = int(len(test_data) * sample_ratio)
       else:
           n = len(test_data)
       
       sampled = random.sample(test_data, n)
       return sampled
   ```

2. **结果标记**：
   - 在结果 JSON 中添加 `fast_mode` 字段
   - 记录采样数量和原始数量
   - 例如：`"fast_mode": {"enabled": true, "sample_size": 472, "total_size": 1888}`

3. **输出提示**：
   ```
   ⚡ Fast 模式已启用
   采样: 472 / 1888 个样本 (25.0%)
   随机种子: 42
   预计评估时间: ~10 分钟
   ```

### 代码修改位置

1. **`run_all_experiments.py` 的 `main()` 函数**：
   - 添加 `--fast`, `--sample_size`, `--sample_ratio` 参数
   - 传递到 `evaluate_experiment` 函数

2. **`evaluate_experiment` 函数**：
   - 在加载测试数据后，根据参数采样
   - 在结果中记录 fast 模式信息

### 使用场景

| 场景 | 参数 | 说明 |
|------|------|------|
| **快速测试** | `--fast` | 默认 472 个样本，快速验证 |
| **调试** | `--sample_size 100` | 极小样本，快速调试 |
| **中等测试** | `--sample_size 1000` | 中等规模测试 |
| **完整评估** | 无参数 | 使用所有样本 |

## 📈 性能对比

| 模式 | 样本数 | 评估时间 | 用途 |
|------|--------|---------|------|
| **Fast** | 472 | ~10 分钟 | 快速测试、调试 |
| **Medium** | 1000 | ~20 分钟 | 中等规模测试 |
| **Full** | 1888 | ~40 分钟 | 完整评估 |

## ⚠️ 注意事项

1. **结果差异**：
   - Fast 模式的结果可能与完整评估有差异
   - 建议在最终报告中使用完整评估结果

2. **可重复性**：
   - 使用固定随机种子（默认 42）
   - 可以通过 `--random_seed` 参数修改

3. **索引构建**：
   - Fast 模式仍然使用完整索引（4504 个文档）
   - 索引构建时间不变（一次性成本）

4. **结果标记**：
   - 所有 fast 模式的结果都会标记
   - 便于区分和对比

## 🔄 与现有 fast_experiment 的关系

- **fast_experiment**: 独立的数据集，包含 472 个样本
- **Fast 模式**: 从完整数据集中采样，更灵活
- **建议**: Fast 模式可以替代 fast_experiment，更灵活且不需要维护两套数据

## 📝 实现优先级

1. **高优先级**：
   - 添加 `--fast` 参数（默认 472 个样本）
   - 实现采样逻辑
   - 结果标记

2. **中优先级**：
   - 添加 `--sample_size` 和 `--sample_ratio` 参数
   - 添加 `--random_seed` 参数

3. **低优先级**：
   - 创建 fast 版本的完整数据集（方案2）
   - 性能分析工具

## ✅ 总结

**推荐方案1**：
- 简单、灵活、易实现
- 保持索引完整，避免问题
- 可以灵活调整采样数量
- 结果可追溯

**实现步骤**：
1. 在 `main()` 中添加参数解析
2. 在 `evaluate_experiment()` 中添加采样逻辑
3. 在结果中标记 fast 模式
4. 测试验证

