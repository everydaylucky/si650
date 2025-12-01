# 完整实验实现指南

## 🎯 概述

本指南说明如何运行完整的实验设计中的所有实验。

## 📋 实验列表

### Track 1: Traditional IR Baselines
- `exp_1_1_bm25` - BM25 Baseline
- `exp_1_2_tfidf` - TF-IDF Baseline  
- `exp_1_3_prf` - Query Expansion + BM25

### Track 2: Zero-shot Dense Models
- `exp_2_1_scibert_zs` - SciBERT Zero-shot
- `exp_2_2_specter2_zs` - SPECTER2 Zero-shot
- `exp_2_3_colbert_zs` - ColBERT Zero-shot
- `exp_2_4_crossenc_zs` - Cross-Encoder Zero-shot

### Track 3: Fine-tuned Models
- `exp_3_1_scibert_ft` - SciBERT Fine-tuned
- `exp_3_2_specter2_ft` - SPECTER2 Fine-tuned
- `exp_3_3_crossenc_ft` - Cross-Encoder Fine-tuned

### Track 4: Fusion Methods
- `exp_4_1_rrf_zs` - RRF (Zero-shot)
- `exp_4_2_rrf_ft` - RRF (Fine-tuned)
- `exp_4_3_l2r_zs` - LightGBM L2R (Zero-shot)
- `exp_4_4_l2r_ft` - LightGBM L2R (Fine-tuned)

### Track 5: Multi-Stage Pipeline
- `exp_5_1_pipeline_basic` - Basic Pipeline
- `exp_5_2_pipeline_optimized` - Optimized Pipeline

## 🚀 快速开始

### 步骤1: 生成所有配置文件

```bash
cd /hy-tmp/final_test
conda activate si650

python scripts/create_experiment_configs.py
```

### 步骤2: 运行单个实验

```bash
# 运行BM25 baseline
python scripts/run_all_experiments.py \
    --experiment exp_1_1_bm25 \
    --data_dir data/processed/fast_experiment

# 运行SciBERT zero-shot
python scripts/run_all_experiments.py \
    --experiment exp_2_1_scibert_zs \
    --data_dir data/processed/fast_experiment
```

### 步骤3: 运行整个Track

```bash
# 运行Track 1 (所有baseline)
python scripts/run_all_experiments.py \
    --track 1 \
    --data_dir data/processed/fast_experiment

# 运行Track 2 (所有zero-shot模型)
python scripts/run_all_experiments.py \
    --track 2 \
    --data_dir data/processed/fast_experiment
```

### 步骤4: 运行所有实验

```bash
# 运行所有实验（需要很长时间！）
python scripts/run_all_experiments.py \
    --all \
    --data_dir data/processed/fast_experiment
```

## 📊 查看结果

### 查看所有实验结果

```bash
python scripts/analyze_results.py
```

### 对比特定实验

```bash
python scripts/analyze_results.py \
    --compare exp_2_1_scibert_zs exp_3_1_scibert_ft
```

### 按模型类型筛选

```bash
python scripts/analyze_results.py \
    --model_type scibert
```

### 按variant筛选

```bash
# 查看所有zero-shot结果
python scripts/analyze_results.py --variant zero-shot

# 查看所有fine-tuned结果
python scripts/analyze_results.py --variant fine-tuned
```

## 🔧 训练模型

### 训练SciBERT

```bash
python scripts/train_scibert.py \
    --config config/fast_experiment_config.yaml
```

### 训练Cross-Encoder

```bash
python scripts/train_cross_encoder.py \
    --config config/fast_experiment_config.yaml
```

### 训练LightGBM L2R

```bash
python scripts/train_l2r.py \
    --train_file data/processed/fast_experiment/train.json
```

## 📁 结果存储

所有实验结果存储在：

```
experiments/
├── results/
│   ├── all_experiments.json          # 所有实验记录
│   ├── experiment_summary.csv         # 实验摘要表格
│   ├── {experiment_id}.json          # 单个实验详细结果
│   └── analysis_report.md            # 分析报告
├── checkpoints/
│   ├── scibert/                      # SciBERT模型
│   ├── cross_encoder/                # Cross-Encoder模型
│   └── l2r/                          # L2R模型
```

## 📈 结果格式

每个实验结果包含：

```json
{
  "experiment_id": "scibert_fine-tuned_20241130_093000",
  "experiment_name": "SciBERT Fine-tuned",
  "model_type": "scibert",
  "variant": "fine-tuned",
  "timestamp": "2024-11-30T09:30:00",
  "metrics": {
    "mrr": 0.273,
    "recall@5": 0.369,
    "recall@10": 0.479,
    ...
  },
  "training_info": {
    "model_path": "experiments/checkpoints/scibert",
    "status": "completed"
  },
  "config": {...}
}
```

## 🎯 实验执行顺序建议

### 推荐顺序

1. **Track 1** (Baselines) - 建立baseline
2. **Track 2** (Zero-shot) - 评估预训练模型
3. **Track 3** (Fine-tuned) - 训练并评估fine-tuned模型
4. **Track 4** (Fusion) - 融合方法
5. **Track 5** (Pipeline) - 完整管道

### 并行执行

可以并行运行：
- Track 1的所有实验（无需训练）
- Track 2的所有实验（无需训练）
- 不同模型的训练（如果有多个GPU）

## ⏱️ 时间估算

| Track | 实验数 | 预计时间 | 说明 |
|-------|--------|----------|------|
| Track 1 | 3 | 1-2小时 | 无需训练 |
| Track 2 | 4 | 2-3小时 | 无需训练 |
| Track 3 | 3 | 12-15小时 | 需要训练 |
| Track 4 | 4 | 3-4小时 | 部分需要训练 |
| Track 5 | 2 | 1-2小时 | 使用已有模型 |
| **总计** | **16** | **19-26小时** | |

## 💡 提示

1. **先运行baseline** - 建立性能基准
2. **保存中间结果** - 每个实验结果自动保存
3. **使用快速数据集** - 先用25%数据验证流程
4. **检查GPU** - 训练需要GPU，评估可以CPU
5. **监控资源** - 注意内存和磁盘空间

## 🐛 故障排除

### 实验失败

检查：
- 数据文件是否存在
- 配置文件是否正确
- 模型路径是否正确
- GPU内存是否足够

### 结果不一致

- 检查随机种子
- 确认使用相同的数据集
- 验证模型版本

## 📚 相关文档

- `TRAINING_GUIDE.md` - 详细训练指南
- `QUICK_REFERENCE.md` - 快速参考
- `EXPERIMENT_RESULTS_ANALYSIS.md` - 结果分析

