#!/usr/bin/env python3
"""
从 data/full 中提取 context_before/after，添加到 fast_experiment 数据中
"""
import json
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_context_mapping(full_data_file: Path) -> dict:
    """从 data/full 创建 context 映射"""
    print("=" * 80)
    print("创建 Context 映射")
    print("=" * 80)
    
    with open(full_data_file) as f:
        full_data = json.load(f)
    
    if isinstance(full_data, dict) and 'samples' in full_data:
        full_samples = full_data['samples']
    elif isinstance(full_data, list):
        full_samples = full_data
    else:
        raise ValueError(f"未知数据格式: {type(full_data)}")
    
    print(f"加载 data/full: {len(full_samples)} 个样本")
    
    # 创建映射：使用 source_paper_id + citation_context 文本作为 key
    context_map = {}
    
    for sample in tqdm(full_samples, desc="处理样本"):
        # 提取 source_paper_id
        source_paper = sample.get('source_paper', {})
        if isinstance(source_paper, dict):
            source_id = source_paper.get('arxiv_id', '')
        else:
            source_id = sample.get('source_paper_id', '')
        
        # 提取 citation_context
        citation_ctx = sample.get('citation_context', {})
        if isinstance(citation_ctx, dict):
            citation_text = citation_ctx.get('text', '')
            context_before = citation_ctx.get('context_before', '')
            context_after = citation_ctx.get('context_after', '')
        else:
            citation_text = str(citation_ctx)
            context_before = ''
            context_after = ''
        
        if source_id and citation_text:
            # 使用 source_id + citation_text 的前100字符作为 key
            key = f"{source_id}:{citation_text[:100]}"
            context_map[key] = {
                'context_before': context_before,
                'context_after': context_after,
                'text': citation_text
            }
    
    print(f"✓ 创建了 {len(context_map)} 个 context 映射")
    return context_map

def add_context_to_fast_experiment(fast_data_file: Path, context_map: dict, output_file: Path):
    """将 context 添加到 fast_experiment 数据中"""
    print("\n" + "=" * 80)
    print("添加 Context 到 fast_experiment")
    print("=" * 80)
    
    with open(fast_data_file) as f:
        fast_data = json.load(f)
    
    print(f"加载 fast_experiment: {len(fast_data)} 个样本")
    
    matched_count = 0
    unmatched_count = 0
    
    for sample in tqdm(fast_data, desc="处理样本"):
        # 提取 source_paper_id
        source_id = sample.get('source_paper_id', '')
        
        # 提取 citation_context
        citation_context = sample.get('citation_context', '')
        if isinstance(citation_context, str):
            citation_text = citation_context
        elif isinstance(citation_context, dict):
            citation_text = citation_context.get('text', '')
        else:
            citation_text = str(citation_context)
        
        # 尝试匹配
        if source_id and citation_text:
            # 尝试精确匹配
            key = f"{source_id}:{citation_text[:100]}"
            context_info = context_map.get(key)
            
            # 如果精确匹配失败，尝试只匹配 source_id（可能有多个 citation）
            if not context_info:
                # 查找所有匹配的 context
                matching_keys = [k for k in context_map.keys() if k.startswith(f"{source_id}:")]
                if matching_keys:
                    # 使用第一个匹配（或者可以尝试文本相似度匹配）
                    key = matching_keys[0]
                    context_info = context_map[key]
            
            if context_info:
                # 将 citation_context 转换为字典格式
                sample['citation_context'] = {
                    'text': citation_text,
                    'context_before': context_info['context_before'],
                    'context_after': context_info['context_after']
                }
                matched_count += 1
            else:
                # 如果没有匹配，保持原样但转换为字典格式
                sample['citation_context'] = {
                    'text': citation_text,
                    'context_before': '',
                    'context_after': ''
                }
                unmatched_count += 1
        else:
            # 如果没有 source_id 或 citation_text，保持原样
            if isinstance(citation_context, str):
                sample['citation_context'] = {
                    'text': citation_context,
                    'context_before': '',
                    'context_after': ''
                }
            unmatched_count += 1
    
    print(f"\n匹配统计:")
    print(f"  匹配成功: {matched_count}/{len(fast_data)} ({matched_count/len(fast_data)*100:.1f}%)")
    print(f"  未匹配: {unmatched_count}/{len(fast_data)} ({unmatched_count/len(fast_data)*100:.1f}%)")
    
    # 保存结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fast_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 已保存到: {output_file}")
    
    return fast_data

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='从 data/full 提取 context 添加到 fast_experiment')
    parser.add_argument('--fast_file', type=str, default='data/processed/fast_experiment/test.json',
                        help='fast_experiment 测试文件')
    parser.add_argument('--full_file', type=str, default='data/full/test.json',
                        help='data/full 测试文件（包含 context）')
    parser.add_argument('--output_file', type=str, default='data/processed/fast_experiment/test_with_context.json',
                        help='输出文件')
    
    args = parser.parse_args()
    
    fast_file = project_root / args.fast_file
    full_file = project_root / args.full_file
    output_file = project_root / args.output_file
    
    if not fast_file.exists():
        print(f"❌ fast_experiment 文件不存在: {fast_file}")
        return
    
    if not full_file.exists():
        print(f"❌ data/full 文件不存在: {full_file}")
        return
    
    # 创建 context 映射
    context_map = create_context_mapping(full_file)
    
    # 添加 context 到 fast_experiment
    add_context_to_fast_experiment(fast_file, context_map, output_file)
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"\n现在可以使用带 context 的数据:")
    print(f"  --data_dir data/processed/fast_experiment")
    print(f"  (使用 test_with_context.json 替换 test.json)")
    print(f"\n或者直接修改 test.json:")
    print(f"  mv {output_file} {fast_file}")

if __name__ == '__main__':
    main()

