# 后台运行实验指南

## 🚀 快速开始

### 方式1: 使用一键脚本（推荐）

```bash
cd /hy-tmp/final_test
bash scripts/run_experiments_background.sh
```

这个脚本会：
- 自动创建日志目录 `experiments/logs/`
- 使用时间戳命名日志文件
- 在后台启动所有实验
- 保存进程ID到文件

### 方式2: 手动使用nohup

```bash
cd /hy-tmp/final_test

# 创建日志目录
mkdir -p experiments/logs

# 后台运行（带时间戳的日志文件）
nohup python scripts/run_all_experiments.py \
    --all \
    --data_dir data/processed/fast_experiment \
    > experiments/logs/experiments_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 记录进程ID
echo $! > experiments/logs/experiments.pid
```

### 方式3: 使用screen（推荐用于长时间运行）

```bash
cd /hy-tmp/final_test

# 创建新的screen会话
screen -S experiments

# 在screen中运行实验
python scripts/run_all_experiments.py --all --data_dir data/processed/fast_experiment

# 按 Ctrl+A 然后 D 来detach（分离会话，实验继续运行）

# 重新连接会话
screen -r experiments

# 查看所有screen会话
screen -ls
```

### 方式4: 使用tmux（推荐用于长时间运行）

```bash
cd /hy-tmp/final_test

# 创建新的tmux会话
tmux new -s experiments

# 在tmux中运行实验
python scripts/run_all_experiments.py --all --data_dir data/processed/fast_experiment

# 按 Ctrl+B 然后 D 来detach

# 重新连接会话
tmux attach -t experiments

# 查看所有tmux会话
tmux ls
```

## 📊 监控实验进度

### 查看实时日志

```bash
# 查看最新的日志文件
tail -f experiments/logs/experiments_*.log

# 或者指定具体文件
tail -f experiments/logs/experiments_20241201_120000.log
```

### 查看进程状态

```bash
# 如果使用了一键脚本，查看PID文件
cat experiments/logs/experiments_*.pid

# 查看进程是否在运行
ps -p $(cat experiments/logs/experiments_*.pid)

# 或者直接查找python进程
ps aux | grep "run_all_experiments.py"
```

### 查看GPU使用情况（如果使用GPU）

```bash
# 查看GPU使用情况
nvidia-smi

# 持续监控GPU
watch -n 1 nvidia-smi
```

### 查看实验进度

```bash
# 查看已完成实验的结果
ls -lh experiments/results/*.json

# 查看实验摘要
python scripts/analyze_results.py

# 查看最新的实验结果
cat experiments/results/COMPREHENSIVE_ANALYSIS.md
```

## 🛑 停止实验

### 如果使用nohup

```bash
# 方法1: 使用PID文件
kill $(cat experiments/logs/experiments_*.pid)

# 方法2: 查找进程并kill
ps aux | grep "run_all_experiments.py" | grep -v grep | awk '{print $2}' | xargs kill

# 方法3: 使用pkill
pkill -f "run_all_experiments.py"
```

### 如果使用screen/tmux

```bash
# Screen: 重新连接后按 Ctrl+C 停止

# Tmux: 重新连接后按 Ctrl+C 停止
```

## 📝 日志管理

### 查看日志文件

```bash
# 列出所有日志文件
ls -lh experiments/logs/

# 查看最新的日志文件
ls -t experiments/logs/*.log | head -1 | xargs tail -f

# 查看日志文件大小
du -h experiments/logs/*.log

# 搜索日志中的错误
grep -i error experiments/logs/*.log

# 搜索特定实验的日志
grep "exp_3_1_scibert_ft" experiments/logs/*.log
```

### 清理旧日志

```bash
# 删除7天前的日志
find experiments/logs/ -name "*.log" -mtime +7 -delete

# 压缩旧日志
find experiments/logs/ -name "*.log" -mtime +1 -exec gzip {} \;
```

## ⚙️ 高级用法

### 只运行特定Track

```bash
# 后台运行Track 3
nohup python scripts/run_all_experiments.py \
    --track 3 \
    --data_dir data/processed/fast_experiment \
    > experiments/logs/track3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 只运行特定实验

```bash
# 后台运行单个实验
nohup python scripts/run_all_experiments.py \
    --experiment exp_3_1_scibert_ft \
    --data_dir data/processed/fast_experiment \
    > experiments/logs/exp_3_1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 设置资源限制

```bash
# 限制CPU使用（使用50%的CPU）
nohup nice -n 10 python scripts/run_all_experiments.py \
    --all \
    --data_dir data/processed/fast_experiment \
    > experiments/logs/experiments_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 限制内存使用（使用ulimit，在脚本开头设置）
ulimit -v 16000000  # 16GB
python scripts/run_all_experiments.py --all --data_dir data/processed/fast_experiment
```

## 📈 预计运行时间

- **所有实验（串行）**: 10-15小时
- **Track 3 (Fine-tuned)**: 10-12小时
- **Track 4 (Fusion)**: 2-3小时
- **Track 5 (Pipeline)**: 1-2小时

## 🔍 故障排查

### 实验没有运行

```bash
# 检查进程是否存在
ps aux | grep "run_all_experiments.py"

# 检查日志文件
tail -100 experiments/logs/experiments_*.log

# 检查是否有错误
grep -i error experiments/logs/experiments_*.log
```

### 实验卡住

```bash
# 查看进程状态
ps aux | grep "run_all_experiments.py"

# 查看GPU使用情况
nvidia-smi

# 查看最新的日志
tail -f experiments/logs/experiments_*.log
```

### 磁盘空间不足

```bash
# 检查磁盘空间
df -h

# 清理临时文件
rm -rf experiments/checkpoints/*/checkpoint-*
rm -rf __pycache__ **/__pycache__
```

## 💡 最佳实践

1. **使用screen或tmux**：更适合长时间运行，可以随时查看和交互
2. **定期检查日志**：确保实验正常运行
3. **保存PID文件**：方便后续管理进程
4. **使用时间戳命名日志**：避免覆盖
5. **定期备份结果**：实验完成后及时备份结果文件

## 📋 完整示例

```bash
cd /hy-tmp/final_test

# 1. 创建日志目录
mkdir -p experiments/logs

# 2. 后台运行所有实验
nohup python scripts/run_all_experiments.py \
    --all \
    --data_dir data/processed/fast_experiment \
    > experiments/logs/experiments_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 3. 记录进程ID
echo $! > experiments/logs/experiments.pid

# 4. 查看日志
tail -f experiments/logs/experiments_*.log

# 5. 检查进度（在另一个终端）
watch -n 60 'ls -lh experiments/results/*.json | wc -l'

# 6. 实验完成后查看结果
python scripts/analyze_results.py
```

