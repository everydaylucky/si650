# 快速参考指南

## 🎯 完整实验流程（3步）

### 步骤1: 准备数据（如果还没有）

```bash
cd /hy-tmp/final_test
conda activate si650

# 创建快速实验数据集（25%数据量）
python scripts/create_fast_dataset.py \
    --train_ratio 0.25 \
    --val_ratio 0.25 \
    --test_ratio 0.25 \
    --output_dir data/processed/fast_experiment
```

### 步骤2: 训练 + 评估（一键完成）

```bash
python scripts/run_full_experiment.py \
    --config config/fast_experiment_config.yaml \
    --data_dir data/processed/fast_experiment
```

### 步骤3: 查看结果

```bash
cat experiments/results/experiment_results.json
```

## 📁 创建的文件

### 训练相关
- ✅ `src/training/__init__.py` - 训练模块初始化
- ✅ `src/training/trainer.py` - SciBERT训练器
- ✅ `scripts/train_scibert.py` - 训练脚本
- ✅ `scripts/run_full_experiment.py` - 完整流程脚本

### 文档
- ✅ `TRAINING_GUIDE.md` - 详细训练指南
- ✅ `QUICK_REFERENCE.md` - 快速参考（本文件）
- ✅ `EXPERIMENT_STATUS.md` - 实验完成情况报告

## 🔧 单独使用训练脚本

### 只训练模型

```bash
python scripts/train_scibert.py \
    --config config/fast_experiment_config.yaml
```

### 只运行评估

```bash
# 使用zero-shot模型
python scripts/run_experiment.py \
    --config config/fast_experiment_config.yaml \
    --data_dir data/processed/fast_experiment

# 使用fine-tuned模型（训练后）
python scripts/run_experiment.py \
    --config config/fast_experiment_config_trained.yaml \
    --data_dir data/processed/fast_experiment
```

## 📊 输出文件位置

- **训练模型**: `experiments/checkpoints/scibert/`
- **评估结果**: `experiments/results/experiment_results.json`
- **更新配置**: `config/fast_experiment_config_trained.yaml`

## ⚡ 常用命令

```bash
# 完整流程（训练+评估）
python scripts/run_full_experiment.py

# 只训练
python scripts/train_scibert.py

# 只评估（zero-shot）
python scripts/run_experiment.py --config config/fast_experiment_config.yaml

# 只评估（fine-tuned）
python scripts/run_experiment.py --config config/fast_experiment_config_trained.yaml
```

## 🎯 预期结果

- **Zero-shot MRR**: ~0.27
- **Fine-tuned MRR**: 0.35-0.40 (预期提升30-50%)
- **训练时间**: 1.5-2小时

## 📝 注意事项

1. 确保已激活conda环境: `conda activate si650`
2. 确保数据文件存在: `data/processed/fast_experiment/train.json`
3. 训练需要GPU（如果没有GPU，会很慢）
4. 训练完成后会自动更新配置文件

## 🐛 问题排查

如果遇到问题，查看：
- `TRAINING_GUIDE.md` - 详细指南和常见问题
- `EXPERIMENT_STATUS.md` - 实验状态报告

