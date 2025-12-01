# 下一步建议

## 当前状态总结

### ✅ 已完成的实验
1. **Track 1: 传统IR基线** - 全部完成
   - BM25 Baseline: MRR = 0.2416
   - TF-IDF Baseline: MRR = 0.2570
   - Query Expansion + BM25: MRR = 0.2499

2. **Track 2: Zero-shot模型** - 部分完成
   - SPECTER2 Zero-shot: MRR = 0.2801 ✅
   - 其他模型需要检查（MRR=0可能有问题）

3. **Track 4: Fusion方法** - 部分完成
   - RRF (Zero-shot): 完成
   - RRF (Fine-tuned): 完成
   - L2R (Zero-shot): 完成
   - L2R (Fine-tuned): 完成

4. **Track 5: Pipeline** - 部分完成
   - Multi-Stage Pipeline (Basic): 完成

### ⚠️ 需要修复/实现的问题

1. **SciBERT训练集成**
   - 当前只是返回路径，没有真正调用训练
   - 需要修复 `train_model` 函数

2. **SPECTER2训练**
   - 尚未实现
   - 可以复用SciBERT的训练方式

3. **Cross-Encoder训练**
   - 脚本已存在，正在运行
   - 需要等待训练完成

4. **Pipeline训练逻辑**
   - 需要实现pipeline类型的训练

5. **MRR=0的问题**
   - 很多实验显示MRR=0，需要检查评估逻辑

## 建议的下一步操作

### 优先级1: 修复SciBERT训练集成 ⭐⭐⭐⭐⭐
```bash
# 修复后运行
python scripts/run_all_experiments.py --experiment exp_3_1_scibert_ft --data_dir data/processed/fast_experiment
```

### 优先级2: 等待Cross-Encoder训练完成 ⭐⭐⭐⭐
- 当前正在训练中（约3 it/s，需要较长时间）
- 训练完成后会自动评估

### 优先级3: 实现SPECTER2训练 ⭐⭐⭐
- 可以复用SciBERT的训练方式
- 或者使用sentence-transformers直接训练

### 优先级4: 检查MRR=0的问题 ⭐⭐⭐
- 检查评估逻辑
- 查看具体哪些实验有问题

### 优先级5: 实现Pipeline训练 ⭐⭐
- Pipeline训练可能需要组合多个模型的训练

## 快速测试建议

如果想快速验证系统，可以：

1. **运行不需要训练的零样本实验**:
```bash
python scripts/run_all_experiments.py --variant zero-shot --data_dir data/processed/fast_experiment
```

2. **运行单个已完成的实验验证**:
```bash
python scripts/run_all_experiments.py --experiment exp_1_1_bm25 --data_dir data/processed/fast_experiment
```

3. **查看结果分析**:
```bash
python scripts/analyze_results.py
```

## 当前最佳结果

- **SPECTER2 Zero-shot**: MRR = 0.2801 （当前最佳）
- **TF-IDF Baseline**: MRR = 0.2570
- **Query Expansion + BM25**: MRR = 0.2499
- **BM25 Baseline**: MRR = 0.2416

