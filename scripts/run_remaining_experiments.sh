#!/bin/bash
# 运行所有剩余实验的脚本

cd "$(dirname "$0")/.." || exit 1

DATA_DIR="data/processed/fast_experiment"

echo "=================================================================================="
echo "运行剩余实验"
echo "=================================================================================="
echo ""

# 剩余实验列表（需要训练的实验）
REMAINING_EXPERIMENTS=(
    "exp_3_1_scibert_ft"           # SciBERT Fine-tuned
    "exp_3_3_crossenc_ft"          # Cross-Encoder Fine-tuned
    "exp_4_2_rrf_ft"               # RRF (Fine-tuned)
    "exp_4_4_l2r_ft"               # LightGBM L2R (Fine-tuned)
    "exp_5_1_pipeline_basic"       # Multi-Stage Pipeline (Basic)
    "exp_5_2_pipeline_optimized"   # Multi-Stage Pipeline (Optimized)
)

# 注意：exp_3_2_specter2_ft 需要SPECTER2训练，但训练脚本尚未实现

echo "将运行以下实验:"
for exp in "${REMAINING_EXPERIMENTS[@]}"; do
    echo "  - $exp"
done
echo ""

# 运行每个实验
for exp_id in "${REMAINING_EXPERIMENTS[@]}"; do
    echo ""
    echo "=================================================================================="
    echo "运行实验: $exp_id"
    echo "=================================================================================="
    echo ""
    
    python scripts/run_all_experiments.py \
        --experiment "$exp_id" \
        --data_dir "$DATA_DIR"
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "⚠ 实验 $exp_id 失败，继续运行下一个..."
    fi
    
    echo ""
done

echo ""
echo "=================================================================================="
echo "所有剩余实验运行完成"
echo "=================================================================================="
echo ""
echo "查看结果:"
echo "  python scripts/analyze_results.py"
echo ""

