#!/usr/bin/env python3
"""
重新组织数据集，使其更便于索引

将 data/full 格式转换为：
1. corpus.json: 包含所有唯一文档（用于索引）
2. test.json: 只包含查询和 ground truth（用于评估）
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def reorganize_data(input_file: Path, output_dir: Path):
    """重新组织数据"""
    print("=" * 80)
    print("重新组织数据集")
    print("=" * 80)
    
    # 加载原始数据
    print(f"\n加载数据: {input_file}")
    with open(input_file) as f:
        raw_data = json.load(f)
    
    if isinstance(raw_data, dict) and 'samples' in raw_data:
        samples = raw_data['samples']
        metadata = raw_data.get('metadata', {})
    elif isinstance(raw_data, list):
        samples = raw_data
        metadata = {}
    else:
        raise ValueError(f"未知数据格式: {type(raw_data)}")
    
    print(f"✓ 加载完成: {len(samples)} 个样本")
    
    # 1. 收集所有唯一文档
    print("\n收集所有唯一文档...")
    all_docs = {}  # arxiv_id -> doc
    doc_count = 0
    
    for sample in tqdm(samples, desc="处理文档"):
        candidates = sample.get('candidates', [])
        for candidate in candidates:
            doc_id = candidate.get('arxiv_id', '')
            if doc_id and doc_id not in all_docs:
                # 处理 categories
                categories = candidate.get('categories', '')
                if isinstance(categories, str):
                    categories = categories.split()
                elif not isinstance(categories, list):
                    categories = []
                
                all_docs[doc_id] = {
                    'id': doc_id,
                    'paper_id': doc_id,
                    'title': candidate.get('title', ''),
                    'abstract': candidate.get('abstract', ''),
                    'categories': categories,
                    'year': candidate.get('year', 2020)
                }
                doc_count += 1
    
    print(f"✓ 收集完成: {len(all_docs)} 个唯一文档")
    
    # 2. 创建简化的测试数据
    print("\n创建简化的测试数据...")
    test_samples = []
    
    for sample in tqdm(samples, desc="处理样本"):
        # 找到 positive candidate
        positive_id = None
        for candidate in sample.get('candidates', []):
            if candidate.get('label', 0) == 1:
                positive_id = candidate.get('arxiv_id', '')
                break
        
        if positive_id:
            # 提取 citation_context
            citation_context = sample.get('citation_context', {})
            if isinstance(citation_context, dict):
                # 保持字典格式（包含 context_before/after）
                pass
            else:
                # 转换为字典格式
                citation_context = {
                    'text': str(citation_context),
                    'context_before': '',
                    'context_after': ''
                }
            
            # 提取 source_paper
            source_paper = sample.get('source_paper', {})
            if source_paper and 'arxiv_id' in source_paper:
                # 转换格式
                categories = source_paper.get('categories', '')
                if isinstance(categories, str):
                    categories = categories.split()
                elif not isinstance(categories, list):
                    categories = []
                
                source_paper = {
                    'id': source_paper.get('arxiv_id', ''),
                    'paper_id': source_paper.get('arxiv_id', ''),
                    'title': source_paper.get('title', ''),
                    'abstract': source_paper.get('abstract', ''),
                    'categories': categories,
                    'year': source_paper.get('year', 2020)
                }
            
            test_samples.append({
                'sample_id': sample.get('sample_id', ''),
                'source_paper': source_paper,
                'citation_context': citation_context,
                'target_paper_id': positive_id
            })
    
    print(f"✓ 创建完成: {len(test_samples)} 个测试样本")
    
    # 3. 保存文件
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存索引文档
    corpus_file = output_dir / "corpus.json"
    corpus_list = list(all_docs.values())
    print(f"\n保存索引文档: {corpus_file}")
    with open(corpus_file, 'w', encoding='utf-8') as f:
        json.dump(corpus_list, f, ensure_ascii=False, indent=2)
    print(f"✓ 已保存: {len(corpus_list)} 个文档")
    
    # 保存测试数据
    test_file = output_dir / "test.json"
    print(f"\n保存测试数据: {test_file}")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    print(f"✓ 已保存: {len(test_samples)} 个样本")
    
    # 保存元数据
    metadata_file = output_dir / "metadata.json"
    metadata_info = {
        'num_samples': len(test_samples),
        'num_documents': len(corpus_list),
        'source_file': str(input_file),
        'reorganized_date': str(Path(__file__).stat().st_mtime)
    }
    if metadata:
        metadata_info.update(metadata)
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_info, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("重新组织完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir}")
    print(f"  - corpus.json: {len(corpus_list)} 个文档（用于索引）")
    print(f"  - test.json: {len(test_samples)} 个样本（用于评估）")
    print(f"  - metadata.json: 元数据信息")
    print(f"\n优势:")
    print(f"  ✓ 索引文件独立，可以复用")
    print(f"  ✓ 测试文件更小，加载更快")
    print(f"  ✓ 清晰的分离：索引 vs 评估")
    print(f"  ✓ 避免重复文档")
    
    return output_dir

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='重新组织数据集')
    parser.add_argument('--input', type=str, default='data/full/test.json',
                        help='输入文件路径')
    parser.add_argument('--output', type=str, default='data/full_indexed',
                        help='输出目录')
    
    args = parser.parse_args()
    
    input_file = project_root / args.input
    output_dir = project_root / args.output
    
    if not input_file.exists():
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    reorganize_data(input_file, output_dir)

if __name__ == '__main__':
    main()

