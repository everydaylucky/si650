# 剩余实验清单

## 📊 当前状态

**已完成**: 14/16 (87.5%)  
**剩余**: 3/16 (18.75%)

## 🔍 剩余实验详情

### 1. SPECTER2 Fine-tuned (exp_3_2_specter2_ft)

**状态**: ❌ 未完成  
**原因**: 训练脚本尚未实现  
**优先级**: ⭐⭐⭐ (中等)

**需要做什么**:
- 实现SPECTER2训练脚本（可以复用SciBERT的训练方式）
- 或使用`sentence-transformers`直接训练SPECTER2模型

**预计时间**: 2-3小时（如果实现训练脚本）

**命令**:
```bash
# 如果实现了训练脚本，运行：
python scripts/run_all_experiments.py --experiment exp_3_2_specter2_ft --data_dir data/processed/fast_experiment
```

---

### 2. Multi-Stage Pipeline (Basic) (exp_5_1_pipeline_basic)

**状态**: ⚠️ 已完成但MRR=0（可能有问题）  
**原因**: 配置中所有阶段都未启用，导致没有检索结果  
**优先级**: ⭐⭐⭐⭐ (高)

**问题分析**:
- 配置文件 `config/experiments/exp_5_1_pipeline_basic.yaml` 中所有 `use_*` 都是 `false`
- 系统会在运行时自动修复，但可能修复不完整
- 需要检查Pipeline Basic的配置逻辑

**需要做什么**:
1. 检查Pipeline Basic的配置是否正确启用各阶段
2. 确保Stage1、Stage2、Stage3都有启用的组件
3. 重新运行实验验证

**命令**:
```bash
# 重新运行Pipeline Basic
python scripts/run_all_experiments.py --experiment exp_5_1_pipeline_basic --data_dir data/processed/fast_experiment
```

---

### 3. Multi-Stage Pipeline (Optimized) (exp_5_2_pipeline_optimized)

**状态**: ❌ 未完成  
**原因**: 依赖fine-tuned模型（现在Cross-Encoder Fine-tuned已完成）  
**优先级**: ⭐⭐⭐⭐⭐ (最高)

**依赖关系**:
- ✅ SciBERT Fine-tuned - 已完成
- ✅ Cross-Encoder Fine-tuned - 已完成
- ❌ SPECTER2 Fine-tuned - 未完成（可选）

**需要做什么**:
- 现在可以运行了！Cross-Encoder Fine-tuned已经完成
- 系统会自动检查并启用必要的fine-tuned模型

**命令**:
```bash
# 运行Pipeline Optimized
python scripts/run_all_experiments.py --experiment exp_5_2_pipeline_optimized --data_dir data/processed/fast_experiment
```

---

## 🚀 推荐运行顺序

### 优先级1: Pipeline Optimized（立即可运行）⭐⭐⭐⭐⭐

```bash
python scripts/run_all_experiments.py --experiment exp_5_2_pipeline_optimized --data_dir data/processed/fast_experiment
```

**原因**: 
- 所有依赖的模型都已训练完成
- 这是最重要的实验之一（完整的多阶段Pipeline）
- 预计性能最好

### 优先级2: Pipeline Basic（修复后运行）⭐⭐⭐⭐

```bash
# 先检查配置，然后运行
python scripts/run_all_experiments.py --experiment exp_5_1_pipeline_basic --data_dir data/processed/fast_experiment
```

**原因**:
- 需要验证配置是否正确
- 是Pipeline Optimized的对比基线

### 优先级3: SPECTER2 Fine-tuned（可选）⭐⭐⭐

**需要先实现训练脚本**，然后运行：
```bash
python scripts/run_all_experiments.py --experiment exp_3_2_specter2_ft --data_dir data/processed/fast_experiment
```

**原因**:
- 训练脚本尚未实现
- 可以复用SciBERT的训练方式
- 对Pipeline Optimized不是必需的（可以使用zero-shot SPECTER2）

---

## 📋 快速运行所有剩余实验

### 方式1: 逐个运行（推荐）

```bash
cd /hy-tmp/final_test

# 1. Pipeline Optimized（最重要）
python scripts/run_all_experiments.py --experiment exp_5_2_pipeline_optimized --data_dir data/processed/fast_experiment

# 2. Pipeline Basic（修复配置后）
python scripts/run_all_experiments.py --experiment exp_5_1_pipeline_basic --data_dir data/processed/fast_experiment

# 3. SPECTER2 Fine-tuned（需要先实现训练脚本）
# python scripts/run_all_experiments.py --experiment exp_3_2_specter2_ft --data_dir data/processed/fast_experiment
```

### 方式2: 后台运行

```bash
cd /hy-tmp/final_test

# 后台运行Pipeline Optimized
nohup python scripts/run_all_experiments.py \
    --experiment exp_5_2_pipeline_optimized \
    --data_dir data/processed/fast_experiment \
    > experiments/logs/pipeline_optimized_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 📊 当前最佳结果

- **最佳实验**: Cross-Encoder Fine-tuned
- **MRR**: 0.3118
- **说明**: 刚刚完成的实验，性能很好

**Top 3 实验结果**:
1. Cross-Encoder Fine-tuned: MRR = 0.3118
2. SciBERT Fine-tuned: MRR = 0.3187
3. SPECTER2 Zero-shot: MRR = 0.2822

---

## ✅ 完成后的下一步

当所有实验完成后：

1. **生成最终分析报告**
   ```bash
   python scripts/analyze_results.py
   ```

2. **查看综合分析**
   ```bash
   cat experiments/results/COMPREHENSIVE_ANALYSIS.md
   ```

3. **对比性能基准**
   ```bash
   cat experiments/results/BENCHMARK_COMPARISON.md
   ```

4. **生成实验总结**
   - 所有实验的MRR、Recall@K、NDCG@K对比
   - Zero-shot vs Fine-tuned对比
   - 不同模型类型的性能分析

