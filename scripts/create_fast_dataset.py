#!/usr/bin/env python3
"""
创建快速实验数据集（缩减数据量）
"""
import json
import random
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

def sample_data(data: List[Dict], ratio: float, stratify_by: str = None) -> List[Dict]:
    """
    采样数据
    
    Args:
        data: 原始数据列表
        ratio: 采样比例 (0.0-1.0)
        stratify_by: 分层字段，如 'source_paper_id' 或 'metadata.section'
    """
    if ratio >= 1.0:
        return data
    
    target_size = int(len(data) * ratio)
    
    if stratify_by:
        # 分层采样
        groups = defaultdict(list)
        for sample in data:
            if stratify_by == 'source_paper_id':
                key = sample.get('source_paper_id', 'unknown')
            elif stratify_by == 'section':
                key = sample.get('metadata', {}).get('section', 'unknown')
            else:
                key = 'all'
            groups[key].append(sample)
        
        # 从每个组中按比例采样
        sampled = []
        for group_key, group_data in groups.items():
            group_size = max(1, int(len(group_data) * ratio))
            sampled.extend(random.sample(group_data, min(group_size, len(group_data))))
        
        # 如果采样后数量不够，随机补充
        if len(sampled) < target_size:
            remaining = [s for s in data if s not in sampled]
            needed = target_size - len(sampled)
            if remaining:
                sampled.extend(random.sample(remaining, min(needed, len(remaining))))
        
        return sampled[:target_size]
    else:
        # 简单随机采样
        return random.sample(data, min(target_size, len(data)))

def reduce_negatives(sample: Dict, train_negatives: int = 5, eval_negatives: int = 20) -> Dict:
    """减少负样本数量"""
    new_sample = sample.copy()
    
    if 'negatives' in new_sample:
        negatives = new_sample['negatives']
        # 判断是训练集还是验证/测试集
        if len(negatives) >= 50:  # 验证/测试集
            new_sample['negatives'] = negatives[:eval_negatives]
        else:  # 训练集
            new_sample['negatives'] = negatives[:train_negatives]
    
    return new_sample

def create_fast_dataset(
    input_train: str,
    input_val: str,
    input_test: str,
    output_dir: str,
    train_ratio: float = 0.25,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    train_negatives: int = 5,
    eval_negatives: int = 20,
    random_seed: int = 42
):
    """创建快速实验数据集"""
    random.seed(random_seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"创建快速实验数据集...")
    print(f"采样比例: 训练集={train_ratio:.0%}, 验证集={val_ratio:.0%}, 测试集={test_ratio:.0%}")
    print(f"负样本: 训练集={train_negatives}, 验证/测试集={eval_negatives}")
    
    # 加载原始数据
    print(f"\n加载原始数据...")
    with open(input_train, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(input_val, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open(input_test, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"原始数据: 训练集={len(train_data)}, 验证集={len(val_data)}, 测试集={len(test_data)}")
    
    # 采样数据（按source_paper_id分层，保持论文级别split）
    print(f"\n采样数据...")
    train_sampled = sample_data(train_data, train_ratio, stratify_by='source_paper_id')
    val_sampled = sample_data(val_data, val_ratio, stratify_by='source_paper_id')
    test_sampled = sample_data(test_data, test_ratio, stratify_by='source_paper_id')
    
    print(f"采样后: 训练集={len(train_sampled)}, 验证集={len(val_sampled)}, 测试集={len(test_sampled)}")
    
    # 减少负样本
    print(f"\n减少负样本...")
    train_fast = [reduce_negatives(s, train_negatives, eval_negatives) for s in train_sampled]
    val_fast = [reduce_negatives(s, train_negatives, eval_negatives) for s in val_sampled]
    test_fast = [reduce_negatives(s, train_negatives, eval_negatives) for s in test_sampled]
    
    # 保存
    print(f"\n保存快速实验数据集...")
    with open(output_path / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_fast, f, indent=2, ensure_ascii=False)
    with open(output_path / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val_fast, f, indent=2, ensure_ascii=False)
    with open(output_path / 'test.json', 'w', encoding='utf-8') as f:
        json.dump(test_fast, f, indent=2, ensure_ascii=False)
    
    # 统计信息
    print(f"\n✅ 快速实验数据集创建完成!")
    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_fast)} 样本 (原始: {len(train_data)})")
    print(f"  验证集: {len(val_fast)} 样本 (原始: {len(val_data)})")
    print(f"  测试集: {len(test_fast)} 样本 (原始: {len(test_data)})")
    print(f"  训练集负样本: {len(train_fast[0].get('negatives', []))} 个/样本")
    print(f"  验证集负样本: {len(val_fast[0].get('negatives', []))} 个/样本")
    print(f"\n保存位置: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='创建快速实验数据集')
    parser.add_argument('--train_ratio', type=float, default=0.25, help='训练集采样比例 (默认: 0.25)')
    parser.add_argument('--val_ratio', type=float, default=0.25, help='验证集采样比例 (默认: 0.25)')
    parser.add_argument('--test_ratio', type=float, default=0.25, help='测试集采样比例 (默认: 0.25)')
    parser.add_argument('--train_negatives', type=int, default=5, help='训练集负样本数量 (默认: 5)')
    parser.add_argument('--eval_negatives', type=int, default=20, help='验证/测试集负样本数量 (默认: 20)')
    parser.add_argument('--output_dir', type=str, default='data/processed/fast_experiment', help='输出目录')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    input_train = project_root / 'data' / 'processed' / 'train.json'
    input_val = project_root / 'data' / 'processed' / 'val.json'
    input_test = project_root / 'data' / 'processed' / 'test.json'
    output_dir = project_root / args.output_dir
    
    create_fast_dataset(
        str(input_train),
        str(input_val),
        str(input_test),
        str(output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        train_negatives=args.train_negatives,
        eval_negatives=args.eval_negatives,
        random_seed=args.random_seed
    )

if __name__ == '__main__':
    main()

