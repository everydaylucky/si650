# Context 实验 Fast 模式运行命令

## 🚀 快速运行命令

### 单个实验运行

```bash
# 实验 1: Query Enhancement (Exp 6.1)
python scripts/run_all_experiments.py --experiment exp_6_1_query_enhancement --data_dir data/full_indexed --fast

# 实验 2: Context Enhancement - Before (Exp 6.1b.1)
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/full_indexed --fast

# 实验 3: Context Enhancement - After (Exp 6.1b.2)
python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir data/full_indexed --fast

# 实验 4: Context Enhancement - Both (Exp 6.1b.3)
python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir data/full_indexed --fast
```

## 📊 Fast 模式说明

- **采样数量**: 默认 472 个样本（约 25%）
- **随机种子**: 42（确保可重复性）
- **评估时间**: 约 10-15 分钟（vs 完整评估 40 分钟）
- **索引**: 使用完整索引（4504 个文档）

## 🔄 批量运行（后台）

```bash
# 所有 Context 实验（后台运行）
nohup python scripts/run_all_experiments.py --experiment exp_6_1_query_enhancement --data_dir data/full_indexed --fast > exp_6_1_query_enhancement_fast.log 2>&1 &

nohup python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/full_indexed --fast > exp_6_1b_1_context_before_fast.log 2>&1 &

nohup python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir data/full_indexed --fast > exp_6_1b_2_context_after_fast.log 2>&1 &

nohup python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir data/full_indexed --fast > exp_6_1b_3_context_both_fast.log 2>&1 &
```

## 📝 其他 Fast 模式选项

### 自定义采样数量

```bash
# 采样 500 个样本
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/full_indexed --sample_size 500

# 采样 25% 的样本
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/full_indexed --sample_ratio 0.25
```

### 自定义随机种子

```bash
# 使用不同的随机种子
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/full_indexed --fast --random_seed 123
```

## 📈 结果说明

Fast 模式的结果会在 JSON 文件中包含 `fast_mode` 字段：

```json
{
  "fast_mode": {
    "enabled": true,
    "sample_size": 472,
    "original_size": 1888,
    "sample_ratio": 0.25,
    "random_seed": 42
  },
  "metrics": {
    "mrr": 0.xxx,
    ...
  }
}
```

## ⚠️ 注意事项

1. **结果差异**: Fast 模式的结果可能与完整评估有差异
2. **最终报告**: 建议在最终报告中使用完整评估结果
3. **可重复性**: 使用固定随机种子（42）确保结果可重复

## 🔍 检查运行状态

```bash
# 查看后台进程
ps aux | grep run_all_experiments

# 查看日志
tail -f exp_6_1b_1_context_before_fast.log

# 查看结果
ls -lh experiments/results/*context*fast*.json
```

