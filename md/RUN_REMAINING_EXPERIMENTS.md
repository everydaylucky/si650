# 运行剩余实验指南

## 📋 剩余实验列表

根据当前实验结果，还有以下实验需要运行：

1. **exp_3_1_scibert_ft** - SciBERT Fine-tuned ⚠️ 需要训练（约3-4小时）
2. **exp_3_3_crossenc_ft** - Cross-Encoder Fine-tuned ⚠️ 需要训练（约5-6小时）
3. **exp_4_2_rrf_ft** - RRF (Fine-tuned) ⚠️ 需要fine-tuned模型
4. **exp_4_4_l2r_ft** - LightGBM L2R (Fine-tuned) ⚠️ 需要训练（约1-2小时）
5. **exp_5_1_pipeline_basic** - Multi-Stage Pipeline (Basic) ✅ 不需要训练
6. **exp_5_2_pipeline_optimized** - Multi-Stage Pipeline (Optimized) ⚠️ 需要训练

**注意**: `exp_3_2_specter2_ft` 需要SPECTER2训练，但训练脚本尚未实现，暂时跳过。

## 🚀 运行方式

### 方式1: 一键运行所有剩余实验（推荐）

```bash
cd /hy-tmp/final_test
bash scripts/run_remaining_experiments.sh
```

### 方式2: 使用Python脚本运行所有实验

```bash
cd /hy-tmp/final_test
python scripts/run_all_experiments.py --all --data_dir data/processed/fast_experiment
```

**注意**: 这会运行所有实验（包括已完成的），但已完成的实验会快速跳过。

### 方式3: 按Track运行

```bash
# 运行Track 3 (Fine-tuned模型) - 需要训练，耗时较长
cd /hy-tmp/final_test
python scripts/run_all_experiments.py --track 3 --data_dir data/processed/fast_experiment

# 运行Track 4 (Fusion方法) - 部分需要训练
python scripts/run_all_experiments.py --track 4 --data_dir data/processed/fast_experiment

# 运行Track 5 (Pipeline) - 部分需要训练
python scripts/run_all_experiments.py --track 5 --data_dir data/processed/fast_experiment
```

### 方式4: 逐个运行（推荐用于调试）

```bash
cd /hy-tmp/final_test

# 1. SciBERT Fine-tuned (需要训练)
python scripts/run_all_experiments.py --experiment exp_3_1_scibert_ft --data_dir data/processed/fast_experiment

# 2. Cross-Encoder Fine-tuned (需要训练)
python scripts/run_all_experiments.py --experiment exp_3_3_crossenc_ft --data_dir data/processed/fast_experiment

# 3. RRF (Fine-tuned) - 需要先完成fine-tuned模型
python scripts/run_all_experiments.py --experiment exp_4_2_rrf_ft --data_dir data/processed/fast_experiment

# 4. L2R (Fine-tuned) (需要训练)
python scripts/run_all_experiments.py --experiment exp_4_4_l2r_ft --data_dir data/processed/fast_experiment

# 5. Pipeline Basic (不需要训练)
python scripts/run_all_experiments.py --experiment exp_5_1_pipeline_basic --data_dir data/processed/fast_experiment

# 6. Pipeline Optimized (需要训练)
python scripts/run_all_experiments.py --experiment exp_5_2_pipeline_optimized --data_dir data/processed/fast_experiment
```

## ⏱️ 预计时间

- **SciBERT Fine-tuned**: 3-4小时
- **Cross-Encoder Fine-tuned**: 5-6小时（当前可能正在训练中）
- **L2R Fine-tuned**: 1-2小时
- **RRF Fine-tuned**: 几分钟（需要先有fine-tuned模型）
- **Pipeline Basic**: 几分钟
- **Pipeline Optimized**: 取决于依赖的模型训练时间

**总预计时间**: 10-15小时（如果串行运行）

## 💡 建议运行顺序

### 快速验证（不需要训练）
```bash
# 先运行不需要训练的实验
python scripts/run_all_experiments.py --experiment exp_5_1_pipeline_basic --data_dir data/processed/fast_experiment
```

### 完整运行（需要训练）
```bash
# 方式1: 使用一键脚本（推荐）
bash scripts/run_remaining_experiments.sh

# 方式2: 使用Python脚本
python scripts/run_all_experiments.py --all --data_dir data/processed/fast_experiment
```

## 📊 运行后查看结果

```bash
# 查看所有实验结果
python scripts/analyze_results.py

# 查看综合分析报告
cat experiments/results/COMPREHENSIVE_ANALYSIS.md

# 查看性能对比
cat experiments/results/BENCHMARK_COMPARISON.md
```

## ⚠️ 注意事项

1. **训练时间**: Fine-tuned模型需要较长时间训练，建议在后台运行
2. **GPU资源**: 确保有足够的GPU资源用于训练
3. **磁盘空间**: 确保有足够空间存储模型checkpoints
4. **依赖关系**: 
   - `exp_4_2_rrf_ft` 需要先完成fine-tuned模型
   - `exp_5_2_pipeline_optimized` 需要先完成相关模型训练

## 🔧 后台运行（推荐）

如果训练时间较长，建议使用后台运行：

```bash
# 使用nohup后台运行
nohup bash scripts/run_remaining_experiments.sh > experiments/run_log.txt 2>&1 &

# 查看运行日志
tail -f experiments/run_log.txt
```

或者使用tmux/screen：

```bash
# 使用tmux
tmux new -s experiments
bash scripts/run_remaining_experiments.sh
# 按 Ctrl+B 然后 D 来detach
# 使用 tmux attach -t experiments 重新连接
```

