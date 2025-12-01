#!/usr/bin/env python3
"""
训练Cross-Encoder模型
"""
import sys
import argparse
import yaml
import json
import traceback
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

def load_training_data(train_file: str):
    """加载训练数据"""
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for sample in tqdm(data, desc="加载数据"):
        # 处理citation_context可能是字典的情况
        citation_context = sample.get("citation_context", "")
        if isinstance(citation_context, dict):
            query = citation_context.get("text", "")
        else:
            query = str(citation_context)
        
        positive = sample.get("target_paper", {})
        negatives = sample.get("negatives", []) or []
        
        if not query or not positive:
            continue
        
        # 格式化正样本
        pos_text = f"{positive.get('title', '')} {positive.get('abstract', '')}"
        
        # 正样本对
        examples.append(InputExample(
            texts=[query, pos_text],
            label=1.0
        ))
        
        # 负样本对（1:3比例）
        for neg in negatives[:3]:
            if isinstance(neg, dict):
                neg_text = f"{neg.get('title', '')} {neg.get('abstract', '')}"
                examples.append(InputExample(
                    texts=[query, neg_text],
                    label=0.0
                ))
    
    return examples

def main():
    parser = argparse.ArgumentParser(description='训练Cross-Encoder模型')
    parser.add_argument('--config', type=str, default='config/fast_experiment_config.yaml')
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--val_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='cross-encoder/ms-marco-MiniLM-L-12-v2')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = project_root / args.config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # 确定数据路径
    if args.train_file:
        train_file = Path(args.train_file)
    else:
        data_dir = project_root / "data" / "processed" / "fast_experiment"
        train_file = data_dir / "train.json"
    
    if not train_file.exists():
        print(f"❌ 训练文件不存在: {train_file}")
        return
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "experiments" / "checkpoints" / "cross_encoder"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练参数
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    
    print("=" * 60)
    print("训练Cross-Encoder模型")
    print(f"模型: {args.model_name}")
    print(f"训练轮次: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print("=" * 60)
    
    try:
        # 加载模型
        print("\n加载预训练模型...")
        model = CrossEncoder(args.model_name)
        print("✓ 模型加载成功")
        
        # 加载数据
        print("\n加载训练数据...")
        train_examples = load_training_data(str(train_file))
        print(f"✓ 训练样本对: {len(train_examples)}")
        
        # 创建数据加载器
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )
        
        # 训练（CrossEncoder使用不同的训练API）
        print("\n开始训练...")
        # CrossEncoder的训练方式：直接使用fit方法
        # 注意：fit方法可能不会自动保存，需要在训练后手动保存
        model.fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            optimizer_params={'lr': learning_rate},
            show_progress_bar=True
        )
        
        # 手动保存模型（fit方法可能不会自动保存）
        print("\n保存模型...")
        model.save(str(output_dir))
        
        # 验证模型文件是否存在
        model_files = ["pytorch_model.bin", "model.safetensors", "config.json", "modules.json"]
        saved_files = [f for f in model_files if (output_dir / f).exists()]
        if saved_files:
            print(f"✓ 模型文件已保存: {', '.join(saved_files)}")
        else:
            print(f"⚠ 警告: 未找到预期的模型文件，但save方法已调用")
        
        print(f"\n✓ 训练完成！模型已保存到: {output_dir}")
        
        # 更新配置
        if config:
            config.setdefault("stage3", {}).setdefault("cross_encoder", {})["fine_tuned_path"] = str(output_dir)
            updated_config_path = project_root / "config" / "fast_experiment_config_trained.yaml"
            with open(updated_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"✓ 配置已更新: {updated_config_path}")
        
    except Exception as e:
        print(f"\n❌ 训练失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

