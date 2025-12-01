# 运行上下文增强实验

## 已完成的修复和实现

### 1. 修复 dense_retriever.py
- ✅ 修复了所有缩进错误
- ✅ 语法检查通过

### 2. 实现上下文增强功能
- ✅ 修改 `multi_stage_pipeline.py` 支持三种模式：
  - `before`: context_before + citation_context
  - `after`: citation_context + context_after
  - `both`: context_before + citation_context + context_after
- ✅ 修改 `run_experiment.py` 提取 context_before/after

### 3. 创建三个实验配置
- ✅ exp_6_1b_1_context_before.yaml
- ✅ exp_6_1b_2_context_after.yaml
- ✅ exp_6_1b_3_context_both.yaml

## 运行实验

### 方法1: 逐个运行
```bash
# 实验1: 仅前文
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/processed/fast_experiment

# 实验2: 仅后文
python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir data/processed/fast_experiment

# 实验3: 前后文
python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir data/processed/fast_experiment
```

### 方法2: 批量运行（后台）
```bash
nohup python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/processed/fast_experiment > exp_6_1b_1.log 2>&1 &
nohup python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir data/processed/fast_experiment > exp_6_1b_2.log 2>&1 &
nohup python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir data/processed/fast_experiment > exp_6_1b_3.log 2>&1 &
```

## 预期结果

| 实验 | 查询组成 | 预期MRR | 基线对比 |
|------|---------|---------|---------|
| 6.1b.1 | context_before + citation | 0.35-0.37 | +2-8% |
| 6.1b.2 | citation + context_after | 0.35-0.37 | +2-8% |
| 6.1b.3 | before + citation + after | 0.35-0.37 | +2-8% |

基线: Pipeline Optimized MRR = 0.3428

## 注意事项

1. **数据格式**: 确保数据中有 context_before 和 context_after
   - 如果当前数据没有，需要从备份数据加载
   - 或者修改数据加载逻辑从原始数据提取

2. **结果对比**: 运行完成后，对比三个实验的结果，看哪个最好

3. **下一步**: 如果效果好，可以在此基础上添加 source_paper
