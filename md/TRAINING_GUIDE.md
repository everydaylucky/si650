# SciBERT训练指南

## 📋 概述

本指南说明如何训练SciBERT模型并用于实验评估。

## 🚀 快速开始

### 方法1: 完整流程（推荐）

运行训练+评估的完整流程：

```bash
cd /hy-tmp/final_test
conda activate si650

python scripts/run_full_experiment.py \
    --config config/fast_experiment_config.yaml \
    --data_dir data/processed/fast_experiment
```

这会自动：
1. 训练SciBERT模型
2. 更新配置文件
3. 使用fine-tuned模型运行评估

### 方法2: 分步执行

#### 步骤1: 训练模型

```bash
python scripts/train_scibert.py \
    --config config/fast_experiment_config.yaml
```

**参数说明**:
- `--config`: 配置文件路径（默认: `config/fast_experiment_config.yaml`）
- `--train_file`: 训练数据文件（默认: 从配置或`data/processed/fast_experiment/train.json`）
- `--val_file`: 验证数据文件（默认: `data/processed/fast_experiment/val.json`）
- `--output_dir`: 模型输出目录（默认: `experiments/checkpoints/scibert`）
- `--epochs`: 训练轮次（覆盖配置）
- `--batch_size`: 批次大小（覆盖配置）
- `--learning_rate`: 学习率（覆盖配置）

**训练完成后**:
- 模型保存在: `experiments/checkpoints/scibert/`
- 自动生成: `config/fast_experiment_config_trained.yaml`

#### 步骤2: 使用fine-tuned模型评估

```bash
python scripts/run_experiment.py \
    --config config/fast_experiment_config_trained.yaml \
    --data_dir data/processed/fast_experiment
```

## 📊 训练配置

在 `config/fast_experiment_config.yaml` 中配置训练参数：

```yaml
training:
  train_scibert: true
  scibert:
    epochs: 3              # 训练轮次
    batch_size: 16         # 批次大小
    learning_rate: 2e-5    # 学习率
    warmup_steps: 100      # Warmup步数
    early_stopping_patience: 2  # Early stopping耐心值
```

## 📁 输出文件

### 训练输出

```
experiments/
└── checkpoints/
    └── scibert/
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer_config.json
        └── vocab.txt
```

### 配置文件更新

训练完成后会自动生成：
- `config/fast_experiment_config_trained.yaml`
  - 包含 `fine_tuned_path` 指向训练好的模型

## 🔍 训练过程监控

训练过程中会显示：
- 训练进度条
- 每个epoch的损失
- 验证集评估结果（如果有）
- 最佳模型保存提示

## ⚙️ 高级用法

### 只训练不评估

```bash
python scripts/run_full_experiment.py \
    --config config/fast_experiment_config.yaml \
    --train_only
```

### 跳过训练直接评估

```bash
python scripts/run_full_experiment.py \
    --config config/fast_experiment_config_trained.yaml \
    --skip_training
```

### 自定义训练参数

```bash
python scripts/train_scibert.py \
    --config config/fast_experiment_config.yaml \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 3e-5
```

## 🐛 常见问题

### 1. 内存不足

**问题**: `CUDA out of memory`

**解决**:
- 减小 `batch_size`（如从16改为8）
- 减少训练样本数量

### 2. 训练很慢

**问题**: 训练速度慢

**解决**:
- 检查是否使用GPU: `nvidia-smi`
- 减少训练轮次: `--epochs 2`
- 减少验证样本数量

### 3. 模型加载失败

**问题**: 训练后模型无法加载

**解决**:
- 检查模型路径是否正确
- 确认配置文件中的 `fine_tuned_path` 已更新
- 检查模型文件是否完整

## 📈 训练效果对比

### Zero-shot vs Fine-tuned

训练完成后，可以对比：

1. **Zero-shot结果** (使用原始配置):
   ```bash
   python scripts/run_experiment.py \
       --config config/fast_experiment_config.yaml \
       --data_dir data/processed/fast_experiment
   ```

2. **Fine-tuned结果** (使用训练后配置):
   ```bash
   python scripts/run_experiment.py \
       --config config/fast_experiment_config_trained.yaml \
       --data_dir data/processed/fast_experiment
   ```

对比指标：
- MRR (Mean Reciprocal Rank)
- Recall@K
- NDCG@K

## 📝 训练数据格式

训练数据格式（`train.json`）:

```json
[
  {
    "citation_context": "查询文本...",
    "target_paper": {
      "title": "论文标题",
      "abstract": "论文摘要"
    },
    "negatives": [
      {
        "title": "负样本标题",
        "abstract": "负样本摘要"
      }
    ]
  }
]
```

## ✅ 检查清单

训练前检查：
- [ ] 训练数据文件存在
- [ ] 验证数据文件存在（可选）
- [ ] 配置文件正确
- [ ] GPU可用（如果使用GPU）

训练后检查：
- [ ] 模型文件已生成
- [ ] 配置文件已更新
- [ ] 可以使用fine-tuned模型运行评估

## 🎯 预期效果

根据快速实验设计：
- **Zero-shot MRR**: ~0.27
- **Fine-tuned MRR**: 预期提升到 **0.35-0.40**
- **训练时间**: 1.5-2小时（3 epochs, 16 batch size）

## 📚 相关文件

- `src/training/trainer.py` - 训练器实现
- `scripts/train_scibert.py` - 训练脚本
- `scripts/run_full_experiment.py` - 完整流程脚本
- `config/fast_experiment_config.yaml` - 配置文件

