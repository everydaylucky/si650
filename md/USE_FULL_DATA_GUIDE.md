# 使用 data/full 数据运行 Context 实验指南

## 📊 数据格式对比

### data/full 格式（新数据）
```json
{
  "split": "test",
  "metadata": {...},
  "samples": [
    {
      "sample_id": "...",
      "source_paper": {
        "arxiv_id": "...",
        "title": "...",
        "abstract": "...",
        "categories": "cs.CL",
        "year": 2020
      },
      "citation_context": {
        "text": "...",
        "context_before": "...",  // ✅ 有前后文
        "context_after": "...",    // ✅ 有前后文
        "section": "Results"
      },
      "candidates": [
        {"arxiv_id": "...", "label": 1, ...},  // positive
        {"arxiv_id": "...", "label": 0, ...}   // negatives
      ]
    }
  ]
}
```

### data/processed/fast_experiment 格式（旧数据）
```json
[
  {
    "citation_context": "文本",  // ❌ 字符串，没有前后文
    "source_paper_id": "...",
    "target_paper": {...}
  }
]
```

## ✅ 已完成的修改

### 1. 修改 `run_all_experiments.py`
- ✅ 支持两种数据格式（字典+samples 和 列表）
- ✅ 正确提取 context_before 和 context_after
- ✅ 处理不同的 source_paper 格式（arxiv_id vs id）
- ✅ 处理不同的 target_paper 格式（candidates vs target_paper）

### 2. 数据加载逻辑
- ✅ 自动检测数据格式
- ✅ 从 citation_context 字典中提取前后文
- ✅ 转换数据格式以匹配现有代码

## 🚀 运行 Context 实验

### 使用 data/full 数据运行三个实验

```bash
cd /hy-tmp/final_test

# 实验1: 仅前文
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/full

# 实验2: 仅后文
python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir data/full

# 实验3: 前后文
python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir data/full
```

### 批量运行（后台）

```bash
nohup python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/full > exp_6_1b_1.log 2>&1 &
nohup python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir data/full > exp_6_1b_2.log 2>&1 &
nohup python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir data/full > exp_6_1b_3.log 2>&1 &
```

## 📊 数据统计

根据检查，data/full/test.json:
- **总样本数**: 1888
- **有 context_before**: 90% (前100个样本中)
- **有 context_after**: 96% (前100个样本中)
- **数据格式**: 字典，包含 'samples' 列表

## 🎯 预期结果

使用 data/full 数据后，三个实验应该会有**不同的结果**：

| 实验 | 查询组成 | 预期MRR | 说明 |
|------|---------|---------|------|
| 6.1b.1 | context_before + citation | 0.35-0.37 | 前文提供背景 |
| 6.1b.2 | citation + context_after | 0.35-0.37 | 后文提供后续信息 |
| 6.1b.3 | before + citation + after | 0.36-0.38 | 完整上下文 |

**基线**: Pipeline Optimized MRR = 0.3428

## ⚠️ 注意事项

1. **数据格式兼容**: 代码已支持两种格式，会自动检测
2. **前后文可用性**: 90% 有前文，96% 有后文，部分样本可能为空
3. **数据量**: data/full 有 1888 个测试样本，比 fast_experiment 多

## 🔍 验证数据加载

运行前可以验证：

```bash
python -c "
from src.utils import load_json
data = load_json('data/full/test.json')
if isinstance(data, dict) and 'samples' in data:
    samples = data['samples']
    sample = samples[0]
    ctx = sample.get('citation_context', {})
    if isinstance(ctx, dict):
        print(f'✓ 数据格式正确')
        print(f'  context_before: {bool(ctx.get(\"context_before\", \"\").strip())}')
        print(f'  context_after: {bool(ctx.get(\"context_after\", \"\").strip())}')
"
```

## 📝 下一步

运行完三个实验后：
1. 对比结果，看哪个最好
2. 如果效果好，在此基础上添加 source_paper
3. 创建 Exp 6.1b.4: 最佳context模式 + source_paper

