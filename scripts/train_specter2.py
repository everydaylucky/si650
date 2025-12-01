#!/usr/bin/env python3
"""
训练SPECTER2模型
"""
import sys
import argparse
import yaml
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.specter2_trainer import SPECTER2Trainer

def main():
    parser = argparse.ArgumentParser(description='训练SPECTER2模型')
    parser.add_argument('--config', type=str, default='config/fast_experiment_config.yaml')
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--val_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    
    args = parser.parse_args()
    
    # 如果提供了命令行参数，直接使用，不检查配置文件标志
    use_config = not (args.train_file or args.output_dir)
    
    if use_config:
        config_path = project_root / args.config
        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        training_config = config.get("training", {})
        if not training_config.get("train_specter2", False):
            print("⚠ 配置中 train_specter2 为 false，跳过训练")
            return
        
        specter2_config = training_config.get("specter2", {})
    else:
        config = {}
        specter2_config = {}
    
    if args.train_file:
        train_file = Path(args.train_file)
    else:
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
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = project_root / "experiments" / "checkpoints" / "specter2"
    
    epochs = args.epochs if args.epochs is not None else int(specter2_config.get("epochs", 3))
    batch_size = args.batch_size if args.batch_size is not None else int(specter2_config.get("batch_size", 16))
    
    if args.learning_rate is not None:
        learning_rate = float(args.learning_rate)
    else:
        lr_config = specter2_config.get("learning_rate", 2e-5)
        if isinstance(lr_config, str):
            learning_rate = float(lr_config)
        else:
            learning_rate = float(lr_config)
    
    warmup_steps = int(specter2_config.get("warmup_steps", 100)) if specter2_config else 100
    early_stopping_patience = int(specter2_config.get("early_stopping_patience", 2)) if specter2_config else 2
    
    print(f"\n训练参数:")
    print(f"  epochs: {epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  warmup_steps: {warmup_steps}")
    print(f"  early_stopping_patience: {early_stopping_patience}")
    
    if use_config and config:
        model_name = config.get("stage1", {}).get("specter2", {}).get("model_name", 
                                                                       "allenai/specter2_base")
    else:
        model_name = "allenai/specter2_base"
    
    trainer = SPECTER2Trainer(model_name=model_name)
    
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
    
    # 只有在使用配置文件时才更新
    if use_config and config:
        print("\n" + "=" * 60)
        print("更新配置文件...")
        config.setdefault("stage1", {}).setdefault("specter2", {})["fine_tuned_path"] = model_path
        
        updated_config_path = project_root / "config" / "fast_experiment_config_trained.yaml"
        with open(updated_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ 配置已更新: {updated_config_path}")
        print(f"   fine_tuned_path: {model_path}")
        print("=" * 60)

if __name__ == "__main__":
    main()
