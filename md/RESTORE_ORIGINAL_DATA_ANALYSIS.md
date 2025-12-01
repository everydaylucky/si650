# 恢复到原始数据格式分析

## 📊 当前情况

### 1. 数据格式变化

**原始格式** (test.json.backup):
```json
{
  "citation_context": "It uses data in languages from Wikipedia...",
  "source_paper_id": "2010.11934",
  ...
}
```

**当前格式** (test.json):
```json
{
  "citation_context": {
    "text": "It uses data in languages from Wikipedia...",
    "context_before": "Many pre-trained versions...",
    "context_after": "We cast all tasks..."
  },
  "source_paper_id": "2010.11934",
  ...
}
```

### 2. 原始实验信息

- **实验ID**: pipeline_optimized_20251201_080642
- **时间**: 2025-12-01 08:06:42
- **MRR**: 0.3428
- **数据集**: data/processed/fast_experiment
- **数据格式**: citation_context 是字符串
- **配置**: 没有 query_enhancement（默认 context_mode="none"）

### 3. 代码兼容性测试

✅ **代码已正确处理两种格式**：
- 如果 `citation_context` 是字符串 → 直接使用
- 如果 `citation_context` 是字典 → 提取 `text` 字段
- 当 `context_mode="none"` 时，两种格式输出相同

## 🔍 问题分析

### 为什么可能影响性能？

1. **理论上不应该影响**：
   - `_build_enhanced_query` 方法正确处理了两种格式
   - 当 `context_mode="none"` 时，只使用 `text` 部分
   - 输出应该完全相同

2. **可能的问题**：
   - 如果代码中某个地方直接使用 `citation_context` 作为字符串
   - 现在它是字典，可能导致类型错误或行为改变
   - 需要检查所有使用 `citation_context` 的地方

3. **数据内容一致性**：
   - ✅ `text` 字段内容与原始字符串相同（521 字符）
   - ✅ 只是格式变化，内容未变

## 🎯 解决方案

### 方案1: 恢复原始数据（推荐）⭐

**如果只想恢复到原来的表现，最简单的方法是恢复原始数据**：

```bash
cd data/processed/fast_experiment
mv test.json.backup test.json
mv train.json.backup train.json
mv val.json.backup val.json
```

**优点**：
- ✅ 完全恢复到原始状态
- ✅ 确保与原始实验结果一致
- ✅ 最简单，无风险

**缺点**：
- ❌ 失去 context 信息
- ❌ 无法运行 Context Enhancement 实验

### 方案2: 保持新格式，确保代码兼容（当前状态）

**代码已经支持两种格式，理论上不应该影响性能**。

**验证方法**：
1. 运行 pipeline_optimized 实验
2. 对比结果是否与原始结果一致（MRR = 0.3428）

**如果结果不一致，可能的原因**：
- 代码中某个地方没有正确处理字典格式
- 需要进一步检查和修复

### 方案3: 创建两个版本的数据

**保持原始数据，同时创建带 context 的版本**：

```bash
# 恢复原始数据
cd data/processed/fast_experiment
mv test.json.backup test.json
mv train.json.backup train.json
mv val.json.backup val.json

# 创建带 context 的版本（用于 Context 实验）
cp test.json test_with_context.json
# 然后运行 add_context_to_fast_experiment.py 生成 test_with_context.json
```

**使用方式**：
- Baseline 实验：使用 `test.json`（原始格式）
- Context 实验：使用 `test_with_context.json`（字典格式）

## 📝 建议

### 对于课程作业

**推荐方案1：恢复原始数据**

原因：
1. 确保 baseline 结果与原始一致（MRR = 0.3428）
2. Context Enhancement 实验可以：
   - 使用 `--fast` 模式从 data/full_indexed 采样
   - 或者创建单独的带 context 的数据文件

### 验证步骤

1. **恢复原始数据**：
   ```bash
   cd data/processed/fast_experiment
   mv test.json.backup test.json
   ```

2. **运行 pipeline_optimized**：
   ```bash
   python scripts/run_all_experiments.py --experiment exp_5_2_pipeline_optimized --data_dir data/processed/fast_experiment
   ```

3. **验证结果**：
   - 如果 MRR ≈ 0.3428 → ✅ 恢复成功
   - 如果 MRR 不同 → ⚠️ 需要进一步检查

## 🔄 数据管理策略

### 推荐的数据组织方式

```
data/processed/fast_experiment/
├── test.json              # 原始格式（用于 baseline）
├── train.json             # 原始格式
├── val.json               # 原始格式
├── test_with_context.json # 带 context（用于 Context 实验）
├── train_with_context.json
└── val_with_context.json
```

**使用方式**：
- Baseline: `--data_dir data/processed/fast_experiment`（使用 test.json）
- Context 实验: 修改代码临时使用 test_with_context.json，或使用 data/full_indexed + --fast

## ⚠️ 注意事项

1. **数据格式一致性**：
   - 确保同一实验使用相同的数据格式
   - 不要混用字符串和字典格式

2. **代码兼容性**：
   - 代码已支持两种格式
   - 但建议统一使用一种格式，避免混淆

3. **实验结果对比**：
   - 只有在相同数据格式下才能公平对比
   - 在报告中说明使用的数据格式

