# 快速实验方案 - 降低时间和成本

## 🎯 目标
在保持大致效果的前提下，将实验时间从15-18小时降低到**5-8小时**，成本降低**60-70%**。

## 📊 优化策略

### 策略1: 数据量缩减 ⭐⭐⭐⭐⭐

**原始数据量**:
- 训练集: 12,587 样本
- 验证集: 1,580 样本
- 测试集: 1,888 样本

**快速实验数据量** (缩减到20-30%):
- 训练集: **2,500-3,000** 样本 (20-25%)
- 验证集: **300-400** 样本 (20-25%)
- 测试集: **400-500** 样本 (20-25%)

**时间节省**: 
- Fine-tuning时间: 3-4小时 → **1-1.5小时** (节省60-70%)
- 实验运行: 2-3小时 → **0.5-1小时** (节省70%)

**效果影响**: 
- 预期MRR可能下降2-5% (0.42 → 0.37-0.40)
- 但仍能验证核心假设（zero-shot vs fine-tuned）

---

### 策略2: 模型精简 ⭐⭐⭐⭐

**完整实验** (12个模型):
- Stage 1: BM25, TF-IDF, SPECTER2
- Stage 2: RRF, ColBERT, SciBERT
- Stage 3: Cross-Encoder, L2R

**快速实验** (5-6个核心模型):
- Stage 1: BM25, SPECTER2 (zero-shot) ✅
- Stage 2: RRF ✅
- Stage 3: Cross-Encoder (zero-shot) ✅
- **Fine-tuning**: 只训练1个模型 (SciBERT或SPECTER2) ⭐

**时间节省**:
- 跳过ColBERT训练: 节省6小时
- 跳过SPECTER2 fine-tuning: 节省4小时
- 跳过Cross-Encoder fine-tuning: 节省5-6小时
- **总计节省**: 10-15小时

**效果影响**:
- 仍能完成zero-shot vs fine-tuned对比
- 仍能验证核心研究问题

---

### 策略3: 负样本减少 ⭐⭐⭐

**原始负样本**:
- 训练集: 10个/样本
- 验证/测试集: 99个/样本

**快速实验负样本**:
- 训练集: **5个/样本** (减少50%)
- 验证/测试集: **20-30个/样本** (减少70-80%)

**时间节省**:
- 数据加载和处理: 节省30-40%
- 训练速度: 提升20-30%

**效果影响**:
- 训练效果可能略降，但验证集仍能评估

---

### 策略4: 训练轮次减少 ⭐⭐⭐

**原始配置**:
- SciBERT: 5 epochs
- SPECTER2: 5 epochs
- Cross-Encoder: 3 epochs

**快速实验配置**:
- SciBERT: **2-3 epochs** + early stopping
- 使用更小的learning rate，更快收敛

**时间节省**: 40-50%

---

### 策略5: 跳过部分实验 ⭐⭐

**可以跳过的实验**:
- ❌ ColBERT (实现复杂，训练时间长)
- ❌ L2R特征工程 (需要所有模型先完成)
- ❌ 完整的消融实验
- ❌ 多阶段管道的完整组合

**必须保留的实验**:
- ✅ BM25 baseline
- ✅ SPECTER2 zero-shot
- ✅ 1个模型的fine-tuning (SciBERT或SPECTER2)
- ✅ Cross-Encoder zero-shot
- ✅ 基本评估指标

---

## 🚀 推荐快速实验方案

### 方案A: 最小可行实验 (5-6小时) ⭐⭐⭐⭐⭐

**数据量**: 25% (3,000训练 + 400验证 + 500测试)

**模型**:
1. BM25 (baseline) - 0.5小时
2. SPECTER2 zero-shot - 1小时
3. SciBERT zero-shot - 0.5小时
4. **SciBERT fine-tuned** - 1.5小时 (只训练1个)
5. Cross-Encoder zero-shot - 1小时
6. RRF融合 - 0.5小时
7. 评估 - 1小时

**总时间**: **5-6小时**

**预期效果**: MRR 0.35-0.38 (仍能验证核心假设)

---

### 方案B: 平衡方案 (7-8小时) ⭐⭐⭐⭐

**数据量**: 30% (3,500训练 + 500验证 + 600测试)

**模型**:
1. BM25 + TF-IDF - 0.5小时
2. SPECTER2 zero-shot - 1小时
3. **SPECTER2 fine-tuned** - 2小时
4. SciBERT zero-shot - 0.5小时
5. Cross-Encoder zero-shot - 1小时
6. RRF融合 - 0.5小时
7. 评估 - 1.5小时

**总时间**: **7-8小时**

**预期效果**: MRR 0.37-0.40 (更好的效果)

---

## 📝 实施步骤

### 1. 创建数据采样脚本

```bash
python scripts/create_fast_dataset.py \
    --train_ratio 0.25 \
    --val_ratio 0.25 \
    --test_ratio 0.25 \
    --output_dir data/processed/fast_experiment
```

### 2. 修改配置文件

创建 `config/fast_experiment_config.yaml`，只启用必要模型。

### 3. 运行快速实验

```bash
python scripts/run_fast_experiment.py
```

---

## 💰 成本对比

| 方案 | GPU时间 | 成本(估算) | 效果 | 推荐度 |
|------|---------|-----------|------|--------|
| 完整实验 | 15-18h | 100% | MRR 0.42+ | ⭐⭐⭐ |
| 方案B (平衡) | 7-8h | 45% | MRR 0.37-0.40 | ⭐⭐⭐⭐ |
| 方案A (最小) | 5-6h | 30% | MRR 0.35-0.38 | ⭐⭐⭐⭐⭐ |

---

## ✅ 快速实验检查清单

- [ ] 创建数据采样脚本
- [ ] 生成快速实验数据集 (25-30%)
- [ ] 修改配置文件（只启用核心模型）
- [ ] 运行BM25 baseline
- [ ] 运行SPECTER2 zero-shot
- [ ] Fine-tune 1个模型 (SciBERT或SPECTER2)
- [ ] 运行Cross-Encoder zero-shot
- [ ] 运行RRF融合
- [ ] 评估并生成报告

---

## 🎯 结论

**推荐方案A (最小可行实验)**:
- ✅ 时间: 5-6小时 (节省70%)
- ✅ 成本: 降低70%
- ✅ 仍能验证核心研究问题
- ✅ 可以后续扩展

**如果时间允许，选择方案B**:
- ✅ 时间: 7-8小时 (节省55%)
- ✅ 效果更好 (MRR 0.37-0.40)
- ✅ 更完整的对比

