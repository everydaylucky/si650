# 快速实验指南 - 5-6小时完成

## 🎯 目标
在保持大致效果的前提下，将实验时间从15-18小时降低到**5-6小时**，成本降低**70%**。

## ⚡ 快速开始（3步）

### 步骤1: 创建快速数据集

```bash
cd /Users/Shared/baiduyun/00\ Code/SI650/final_test

# 创建25%数据量的快速实验数据集
python scripts/create_fast_dataset.py \
    --train_ratio 0.25 \
    --val_ratio 0.25 \
    --test_ratio 0.25 \
    --train_negatives 5 \
    --eval_negatives 20 \
    --output_dir data/processed/fast_experiment
```

**结果**:
- 训练集: ~3,000 样本 (原始12,587)
- 验证集: ~400 样本 (原始1,580)
- 测试集: ~500 样本 (原始1,888)
- 负样本: 训练集5个，验证/测试集20个

### 步骤2: 运行快速实验

```bash
# 使用快速实验配置
python scripts/run_experiment.py \
    --config config/fast_experiment_config.yaml \
    --data_dir data/processed/fast_experiment
```

### 步骤3: 查看结果

结果保存在 `experiments/results/fast_experiment_results.json`

---

## 📊 优化效果对比

| 项目 | 完整实验 | 快速实验 | 节省 |
|------|---------|---------|------|
| **数据量** | 16,055样本 | 3,900样本 | 76% ↓ |
| **训练时间** | 12-15小时 | 1.5-2小时 | 85% ↓ |
| **总时间** | 15-18小时 | 5-6小时 | 70% ↓ |
| **GPU成本** | 100% | 30% | 70% ↓ |
| **预期MRR** | 0.42+ | 0.35-0.38 | -10% |

---

## 🔬 实验内容（快速版）

### 必须完成的实验

1. ✅ **BM25 Baseline** (0.5小时)
   - 传统IR基线
   - 无需GPU

2. ✅ **SPECTER2 Zero-shot** (1小时)
   - 预训练模型，无需训练
   - 只需构建索引

3. ✅ **SciBERT Zero-shot** (0.5小时)
   - 快速测试

4. ✅ **SciBERT Fine-tuned** (1.5小时) ⭐
   - **只训练1个模型**
   - 3 epochs (原始5 epochs)
   - 验证zero-shot vs fine-tuned对比

5. ✅ **Cross-Encoder Zero-shot** (1小时)
   - 使用MS-MARCO预训练模型
   - 无需训练

6. ✅ **RRF融合** (0.5小时)
   - 融合BM25和SPECTER2结果

7. ✅ **评估** (1小时)
   - 计算MRR, Recall@K, NDCG@K

### 跳过的实验（节省时间）

- ❌ TF-IDF (节省0.5小时)
- ❌ ColBERT (节省6小时)
- ❌ SPECTER2 Fine-tuning (节省4小时)
- ❌ Cross-Encoder Fine-tuning (节省5-6小时)
- ❌ L2R特征工程 (节省3-4小时)
- ❌ 完整消融实验 (节省2-3小时)

**总计节省**: 20-24小时

---

## 📈 预期结果

### 性能预期

| 模型 | 完整实验MRR | 快速实验MRR | 说明 |
|------|------------|------------|------|
| BM25 | 0.20 | 0.18-0.20 | 基本一致 |
| SPECTER2 (zero) | 0.33 | 0.30-0.33 | 基本一致 |
| SciBERT (zero) | 0.28 | 0.26-0.28 | 基本一致 |
| SciBERT (FT) | 0.35 | 0.32-0.35 | 略降但可接受 |
| Cross-Encoder (zero) | 0.43 | 0.40-0.43 | 基本一致 |
| RRF | 0.36 | 0.33-0.36 | 基本一致 |

### 核心验证

✅ **仍能验证的核心问题**:
1. Zero-shot vs Fine-tuned对比 (SciBERT)
2. 传统IR vs 神经模型对比
3. 多阶段管道效果
4. 基本评估指标

---

## 💡 进一步优化建议

### 如果时间更紧（3-4小时）

1. **数据量再减半** (12.5%):
   ```bash
   python scripts/create_fast_dataset.py --train_ratio 0.125
   ```

2. **只做zero-shot实验**:
   - 跳过所有fine-tuning
   - 只对比预训练模型
   - 时间: 3-4小时

3. **只训练1个epoch**:
   - SciBERT: 1 epoch + early stopping
   - 时间: 0.5-1小时

### 如果效果不够好

1. **增加数据量到40%**:
   ```bash
   python scripts/create_fast_dataset.py --train_ratio 0.4
   ```

2. **增加训练轮次**:
   - SciBERT: 5 epochs
   - 时间: 2-2.5小时

---

## ✅ 检查清单

- [ ] 创建快速数据集 (`create_fast_dataset.py`)
- [ ] 验证数据格式正确
- [ ] 使用快速实验配置 (`fast_experiment_config.yaml`)
- [ ] 运行BM25 baseline
- [ ] 运行SPECTER2 zero-shot
- [ ] Fine-tune SciBERT (3 epochs)
- [ ] 运行Cross-Encoder zero-shot
- [ ] 运行RRF融合
- [ ] 评估并生成报告

---

## 🎯 结论

**推荐方案**: 25%数据量 + 只训练SciBERT

- ✅ 时间: 5-6小时 (节省70%)
- ✅ 成本: 降低70%
- ✅ 效果: MRR 0.35-0.38 (仍能验证核心假设)
- ✅ 可以后续扩展完整实验

**如果效果不够，可以**:
- 增加到40%数据量 (+2小时)
- 训练5 epochs (+1小时)
- 总时间仍只需8-9小时

---

**快速实验配置**: `config/fast_experiment_config.yaml`  
**数据采样脚本**: `scripts/create_fast_dataset.py`  
**详细计划**: `FAST_EXPERIMENT_PLAN.md`

