# 🚀 快速运行剩余实验

## 最简单的方式（推荐）

```bash
cd /hy-tmp/final_test
python scripts/run_all_experiments.py --all --data_dir data/processed/fast_experiment
```

**说明**: 这个命令会运行所有16个实验，但已完成的实验会快速跳过，只运行剩余的7个实验。

## 其他运行方式

### 方式1: 使用一键脚本
```bash
cd /hy-tmp/final_test
bash scripts/run_remaining_experiments.sh
```

### 方式2: 按Track运行（推荐用于分阶段）

```bash
cd /hy-tmp/final_test

# Track 3: Fine-tuned模型（需要训练，预计10-12小时）
python scripts/run_all_experiments.py --track 3 --data_dir data/processed/fast_experiment

# Track 4: Fusion方法（部分需要训练，预计2-3小时）
python scripts/run_all_experiments.py --track 4 --data_dir data/processed/fast_experiment

# Track 5: Pipeline（部分需要训练）
python scripts/run_all_experiments.py --track 5 --data_dir data/processed/fast_experiment
```

### 方式3: 后台运行（推荐用于长时间训练）

```bash
cd /hy-tmp/final_test

# 使用nohup后台运行
nohup python scripts/run_all_experiments.py --all --data_dir data/processed/fast_experiment > experiments/run_log.txt 2>&1 &

# 查看运行日志
tail -f experiments/run_log.txt
```

## 📋 剩余实验列表

1. **exp_3_1_scibert_ft** - SciBERT Fine-tuned ⚠️ 需要训练（3-4小时）
2. **exp_3_2_specter2_ft** - SPECTER2 Fine-tuned ⚠️ 训练脚本未实现，会跳过
3. **exp_3_3_crossenc_ft** - Cross-Encoder Fine-tuned ⚠️ 需要训练（5-6小时）
4. **exp_4_2_rrf_ft** - RRF (Fine-tuned) ✅ 不需要训练，但需要fine-tuned模型
5. **exp_4_4_l2r_ft** - LightGBM L2R (Fine-tuned) ⚠️ 需要训练（1-2小时）
6. **exp_5_1_pipeline_basic** - Multi-Stage Pipeline (Basic) ✅ 不需要训练
7. **exp_5_2_pipeline_optimized** - Multi-Stage Pipeline (Optimized) ⚠️ 需要fine-tuned模型

## ⏱️ 预计总时间

- **串行运行**: 10-15小时
- **并行运行**（如果有多个GPU）: 5-8小时

## 📊 运行后查看结果

```bash
# 查看所有实验结果
python scripts/analyze_results.py

# 查看综合分析报告
cat experiments/results/COMPREHENSIVE_ANALYSIS.md

# 查看性能对比
cat experiments/results/BENCHMARK_COMPARISON.md
```

