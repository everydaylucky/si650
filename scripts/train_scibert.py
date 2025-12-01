#!/usr/bin/env python3
"""
训练SciBERT模型
"""
import sys
import argparse
import yaml
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import SciBERTTrainer

def main():
    parser = argparse.ArgumentParser(description='训练SciBERT模型')
    parser.add_argument('--config', type=str, default='config/fast_experiment_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--train_file', type=str, default=None,
                        help='训练数据文件路径（覆盖配置）')
    parser.add_argument('--val_file', type=str, default=None,
                        help='验证数据文件路径（覆盖配置）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='模型输出目录（覆盖配置）')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮次（覆盖配置）')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小（覆盖配置）')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='学习率（覆盖配置）')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = project_root / args.config
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    training_config = config.get("training", {})
    if not training_config.get("train_scibert", False):
        print("⚠ 配置中 train_scibert 为 false，跳过训练")
        return
    
    scibert_config = training_config.get("scibert", {})
    
    # 确定数据路径
    if args.train_file:
        train_file = Path(args.train_file)
    else:
        # 从命令行参数或配置推断
        data_dir = project_root / "data" / "processed" / "fast_experiment"
        train_file = data_dir / "train.json"
    
    if args.val_file:
        val_file = Path(args.val_file)
    else:
        data_dir = train_file.parent
        val_file = data_dir / "val.json"
    
    if not train_file.exists():
        print(f"❌ 训练文件不存在: {train_file}")
        return
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = project_root / "experiments" / "checkpoints" / "scibert"
    
    # 训练参数（确保类型正确）
    epochs = args.epochs if args.epochs is not None else int(scibert_config.get("epochs", 3))
    batch_size = args.batch_size if args.batch_size is not None else int(scibert_config.get("batch_size", 16))
    
    # 学习率需要特别处理，因为YAML中的科学计数法可能被解析为字符串
    if args.learning_rate is not None:
        learning_rate = float(args.learning_rate)
    else:
        lr_config = scibert_config.get("learning_rate", 2e-5)
        if isinstance(lr_config, str):
            # 处理字符串形式的学习率（如 "2e-5"）
            learning_rate = float(lr_config)
        else:
            learning_rate = float(lr_config)
    
    warmup_steps = int(scibert_config.get("warmup_steps", 100))
    early_stopping_patience = int(scibert_config.get("early_stopping_patience", 2))
    
    # 验证参数
    print(f"\n训练参数:")
    print(f"  epochs: {epochs} (type: {type(epochs).__name__})")
    print(f"  batch_size: {batch_size} (type: {type(batch_size).__name__})")
    print(f"  learning_rate: {learning_rate} (type: {type(learning_rate).__name__})")
    print(f"  warmup_steps: {warmup_steps}")
    print(f"  early_stopping_patience: {early_stopping_patience}")
    
    # 模型名称
    model_name = config.get("stage2", {}).get("bi_encoder", {}).get("model_name", 
                                                                      "allenai/scibert_scivocab_uncased")
    
    # 创建训练器并训练
    trainer = SciBERTTrainer(model_name=model_name)
    
    model_path = trainer.train(
        train_file=str(train_file),
        val_file=str(val_file) if val_file.exists() else None,
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        early_stopping_patience=early_stopping_patience
    )
    
    # 更新配置文件
    print("\n" + "=" * 60)
    print("更新配置文件...")
    config["stage2"]["bi_encoder"]["fine_tuned_path"] = model_path
    
    # 保存更新后的配置
    updated_config_path = project_root / "config" / "fast_experiment_config_trained.yaml"
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ 配置已更新: {updated_config_path}")
    print(f"   fine_tuned_path: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()

