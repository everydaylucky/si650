#!/bin/bash
# 统一 Fast 模式运行所有实验
# 确保所有实验在完全相同的条件下运行

DATA_DIR="data/processed/fast_experiment"

echo "=================================================================================="
echo "统一 Fast 模式实验运行"
echo "=================================================================================="
echo ""
echo "数据集: $DATA_DIR"
echo "样本数: 472"
echo "负样本比例: 1:20"
echo "评估时间: ~10-15 分钟/实验"
echo ""
echo "=================================================================================="
echo ""

# Baseline: Optimized Pipeline
echo "[1/6] 运行 Baseline: Optimized Pipeline"
echo "--------------------------------------------------------------------------------"
python scripts/run_all_experiments.py --experiment exp_5_2_pipeline_optimized --data_dir $DATA_DIR
if [ $? -ne 0 ]; then
    echo "❌ Baseline 实验失败"
    exit 1
fi
echo ""

# Context Enhancement 实验
echo "[2/6] 运行 Query Enhancement (Exp 6.1)"
echo "--------------------------------------------------------------------------------"
python scripts/run_all_experiments.py --experiment exp_6_1_query_enhancement --data_dir $DATA_DIR
if [ $? -ne 0 ]; then
    echo "⚠ Query Enhancement 实验失败，继续..."
fi
echo ""

echo "[3/6] 运行 Context Enhancement - Before (Exp 6.1b.1)"
echo "--------------------------------------------------------------------------------"
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir $DATA_DIR
if [ $? -ne 0 ]; then
    echo "⚠ Context Before 实验失败，继续..."
fi
echo ""

echo "[4/6] 运行 Context Enhancement - After (Exp 6.1b.2)"
echo "--------------------------------------------------------------------------------"
python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir $DATA_DIR
if [ $? -ne 0 ]; then
    echo "⚠ Context After 实验失败，继续..."
fi
echo ""

echo "[5/6] 运行 Context Enhancement - Both (Exp 6.1b.3)"
echo "--------------------------------------------------------------------------------"
python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir $DATA_DIR
if [ $? -ne 0 ]; then
    echo "⚠ Context Both 实验失败，继续..."
fi
echo ""

echo "=================================================================================="
echo "所有实验完成！"
echo "=================================================================================="
echo ""
echo "结果文件位置: experiments/results/"
echo ""
echo "分析结果:"
echo "  python scripts/analyze_results.py"
echo ""

