# Fast Experiment 数据集 Context 增强指南

## ✅ 已完成

### 1. 数据更新

已成功从 `data/full` 中提取 `context_before` 和 `context_after`，并添加到 `fast_experiment` 数据中：

- ✅ **test.json**: 472 个样本，100% 匹配
- ✅ **train.json**: 3146 个样本，100% 匹配
- ✅ **val.json**: 395 个样本，100% 匹配

### 2. 数据格式

现在 `fast_experiment` 的数据格式为：

```json
{
  "citation_context": {
    "text": "It uses data in languages from Wikipedia...",
    "context_before": "Many pre-trained versions of XLM...",
    "context_after": "We cast all tasks into the text-to-text format..."
  },
  "source_paper_id": "2010.11934",
  "target_paper_id": "1910.07475",
  ...
}
```

### 3. 统计信息

- **有 context_before**: 96.6% (456/472)
- **有 context_after**: 95.1% (449/472)
- **匹配成功率**: 100%

## 🚀 使用方法

### 直接使用（推荐）

现在可以直接使用 `fast_experiment` 数据集，代码会自动识别字典格式的 `citation_context`：

```bash
# 运行 Context Enhancement 实验
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/processed/fast_experiment
python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir data/processed/fast_experiment
python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir data/processed/fast_experiment
```

### 运行所有实验

```bash
# 使用批量脚本
./scripts/run_all_fast_experiments.sh
```

## 📊 优势

### 1. 保持 fast_experiment 的优势
- ✅ 评估速度快（~10-15 分钟）
- ✅ 负样本比例 1:20（对课程作业合理）
- ✅ 样本数 472（足够评估）

### 2. 添加了 Context 信息
- ✅ 96.6% 的样本有 `context_before`
- ✅ 95.1% 的样本有 `context_after`
- ✅ 可以测试 Context Enhancement 的效果

### 3. 数据格式统一
- ✅ 使用字典格式的 `citation_context`
- ✅ 代码自动识别和处理
- ✅ 与 `data/full` 格式兼容

## 🔄 数据恢复

如果需要恢复原始数据：

```bash
cd data/processed/fast_experiment
mv test.json.backup test.json
mv train.json.backup train.json
mv val.json.backup val.json
```

## 📝 代码兼容性

代码已经支持这种格式：

1. **`run_all_experiments.py`**：
   - 自动检测 `citation_context` 是字典还是字符串
   - 自动提取 `context_before` 和 `context_after`

2. **`multi_stage_pipeline.py`**：
   - `_build_enhanced_query` 方法会使用 `context_before` 和 `context_after`

3. **数据加载**：
   - 所有数据加载逻辑都已支持字典格式

## 🎯 实验对比

现在可以在 `fast_experiment` 上公平对比：

| 实验 | 数据集 | Context 信息 | 评估时间 |
|------|--------|-------------|---------|
| Pipeline Optimized | fast_experiment | ❌ | ~10-15 分钟 |
| Context Before | fast_experiment | ✅ context_before | ~10-15 分钟 |
| Context After | fast_experiment | ✅ context_after | ~10-15 分钟 |
| Context Both | fast_experiment | ✅ both | ~10-15 分钟 |

**所有实验条件完全相同，对比最公平！**

## ⚠️ 注意事项

1. **原文件已备份**：
   - `test.json.backup`
   - `train.json.backup`
   - `val.json.backup`

2. **数据格式变化**：
   - `citation_context` 从字符串变为字典
   - 代码已自动处理，无需修改

3. **Context 可用性**：
   - 96.6% 有 `context_before`
   - 95.1% 有 `context_after`
   - 部分样本可能为空（这是正常的）

## 🎉 总结

现在 `fast_experiment` 数据集：
- ✅ 保持了快速评估的优势
- ✅ 添加了 Context 信息
- ✅ 可以公平对比 Context Enhancement 的效果
- ✅ 适合课程作业使用

可以直接运行实验了！

