#!/bin/bash
# 后台运行所有实验的脚本

cd "$(dirname "$0")/.." || exit 1

DATA_DIR="data/processed/fast_experiment"
LOG_DIR="experiments/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/experiments_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/experiments_${TIMESTAMP}.pid"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "=================================================================================="
echo "后台运行所有实验"
echo "=================================================================================="
echo ""
echo "数据目录: $DATA_DIR"
echo "日志文件: $LOG_FILE"
echo "PID文件: $PID_FILE"
echo ""

# 使用nohup在后台运行
nohup python scripts/run_all_experiments.py \
    --all \
    --data_dir "$DATA_DIR" \
    > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!
echo "$PID" > "$PID_FILE"

echo "✓ 实验已在后台启动"
echo "  进程ID: $PID"
echo "  日志文件: $LOG_FILE"
echo "  PID文件: $PID_FILE"
echo ""
echo "查看运行日志:"
echo "  tail -f $LOG_FILE"
echo ""
echo "查看进程状态:"
echo "  ps -p $PID"
echo ""
echo "停止实验:"
echo "  kill $PID"
echo "  或: kill \$(cat $PID_FILE)"
echo ""
echo "查看所有实验日志:"
echo "  ls -lh $LOG_DIR"
echo ""

