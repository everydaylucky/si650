#!/usr/bin/env python3
"""
自动生成所有实验的配置文件
"""
import sys
from pathlib import Path
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.experiment_config import ALL_EXPERIMENTS

def create_config(exp_config, base_config_path: Path):
    """为单个实验创建配置文件"""
    # 加载基础配置
    if base_config_path.exists():
        with open(base_config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # 根据实验类型调整配置
    model_type = exp_config.model_type
    variant = exp_config.variant
    
    # Stage 1配置
    config.setdefault("stage1", {})
    config["stage1"]["use_bm25"] = model_type == "bm25"
    config["stage1"]["use_tfidf"] = model_type == "tfidf"
    config["stage1"]["use_prf"] = model_type == "prf"
    config["stage1"]["use_specter2"] = model_type == "specter2" and variant == "zero-shot"
    config["stage1"]["top_k"] = 1000
    
    # Stage 2配置
    config.setdefault("stage2", {})
    config["stage2"]["use_rrf"] = model_type == "rrf"
    config["stage2"]["use_bi_encoder"] = model_type == "scibert"
    config["stage2"]["use_colbert"] = model_type == "colbert"
    config["stage2"]["top_k"] = 50
    
    # Stage 3配置
    config.setdefault("stage3", {})
    config["stage3"]["use_cross_encoder"] = model_type == "cross_encoder"
    config["stage3"]["use_l2r"] = model_type == "l2r"
    config["stage3"]["top_k"] = 20
    
    # 设置fine-tuned路径
    if variant == "fine-tuned":
        if model_type == "scibert":
            config["stage2"]["bi_encoder"]["fine_tuned_path"] = "experiments/checkpoints/scibert"
        elif model_type == "specter2":
            config["stage1"]["specter2"]["fine_tuned_path"] = "experiments/checkpoints/specter2"
        elif model_type == "cross_encoder":
            config["stage3"]["cross_encoder"]["fine_tuned_path"] = "experiments/checkpoints/cross_encoder"
    
    # 训练配置
    config.setdefault("training", {})
    config["training"]["train_scibert"] = model_type == "scibert" and variant == "fine-tuned"
    config["training"]["train_specter2"] = model_type == "specter2" and variant == "fine-tuned"
    config["training"]["train_cross_encoder"] = model_type == "cross_encoder" and variant == "fine-tuned"
    config["training"]["train_l2r"] = model_type == "l2r"
    
    return config

def main():
    configs_dir = project_root / "config" / "experiments"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    base_config_path = project_root / "config" / "fast_experiment_config.yaml"
    
    print("=" * 80)
    print("生成实验配置文件")
    print("=" * 80)
    
    created = 0
    for exp_id, exp_config in ALL_EXPERIMENTS.items():
        config = create_config(exp_config, base_config_path)
        
        output_path = configs_dir / f"{exp_id}.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ {exp_id:30s} -> {output_path.name}")
        created += 1
    
    print(f"\n✓ 已创建 {created} 个配置文件")
    print(f"  位置: {configs_dir}")

if __name__ == "__main__":
    main()

