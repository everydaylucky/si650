#!/usr/bin/env python3
"""
检查数据格式是否符合要求
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

def check_data_file(file_path: str, dataset_name: str):
    """检查单个数据文件"""
    print(f"\n{'='*60}")
    print(f"检查 {dataset_name}: {file_path}")
    print(f"{'='*60}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"❌ 无法加载文件: {e}")
        return False
    
    # 处理不同的数据格式
    if isinstance(raw_data, dict):
        if 'samples' in raw_data:
            data = raw_data['samples']
            metadata = raw_data.get('metadata', {})
            print(f"\n📋 文件格式: 包含metadata的对象")
            if metadata:
                print(f"  元数据: {metadata}")
        else:
            print(f"❌ 字典格式但缺少'samples'字段")
            return False
    elif isinstance(raw_data, list):
        data = raw_data
        print(f"\n📋 文件格式: JSON数组")
    else:
        print(f"❌ 未知的数据格式: {type(raw_data)}")
        return False
    
    # 基本统计
    total_samples = len(data)
    print(f"\n📊 基本统计:")
    print(f"  总样本数: {total_samples:,}")
    
    if total_samples == 0:
        print("❌ 文件为空")
        return False
    
    # 检查第一个样本的结构
    print(f"\n📋 样本结构检查:")
    sample = data[0]
    print(f"  第一个样本的字段: {list(sample.keys())}")
    
    # 必需字段检查
    required_fields = {
        'citation_context': str,
        'source_paper_id': str,
        'target_paper_id': str,
        'source_paper': dict,
        'target_paper': dict
    }
    
    missing_fields = []
    type_errors = []
    
    for field, expected_type in required_fields.items():
        if field not in sample:
            missing_fields.append(field)
        elif not isinstance(sample[field], expected_type):
            type_errors.append(f"{field}: 期望 {expected_type.__name__}, 实际 {type(sample[field]).__name__}")
    
    if missing_fields:
        print(f"  ❌ 缺少必需字段: {missing_fields}")
    else:
        print(f"  ✅ 所有必需字段存在")
    
    if type_errors:
        print(f"  ❌ 类型错误: {type_errors}")
    else:
        print(f"  ✅ 字段类型正确")
    
    # 检查source_paper和target_paper的结构
    print(f"\n📄 论文对象结构检查:")
    for paper_type in ['source_paper', 'target_paper']:
        if paper_type in sample:
            paper = sample[paper_type]
            paper_required = ['id', 'title', 'abstract']
            paper_missing = [f for f in paper_required if f not in paper]
            
            if paper_missing:
                print(f"  ❌ {paper_type} 缺少字段: {paper_missing}")
            else:
                print(f"  ✅ {paper_type} 结构完整")
                print(f"     字段: {list(paper.keys())}")
    
    # 检查负样本（仅训练集）
    has_negatives = 'negatives' in sample
    if dataset_name == 'train' and not has_negatives:
        print(f"\n  ⚠️  训练集应该包含negatives字段")
    elif has_negatives:
        print(f"\n  ✅ 包含negatives字段")
        if isinstance(sample['negatives'], list):
            print(f"     负样本数量: {len(sample['negatives'])}")
    
    # 批量检查（采样检查）
    print(f"\n🔍 批量质量检查（采样100个）:")
    sample_size = min(100, total_samples)
    samples_to_check = data[:sample_size]
    
    issues = {
        'missing_citation_context': 0,
        'missing_source_paper': 0,
        'missing_target_paper': 0,
        'time_violations': 0,
        'empty_text': 0,
        'short_context': 0
    }
    
    for i, s in enumerate(samples_to_check):
        # 检查必需字段
        if not s.get('citation_context'):
            issues['missing_citation_context'] += 1
        elif len(s['citation_context'].split()) < 10:
            issues['short_context'] += 1
        
        if not s.get('source_paper'):
            issues['missing_source_paper'] += 1
        if not s.get('target_paper'):
            issues['missing_target_paper'] += 1
        
        # 检查时间一致性
        source_year = s.get('source_paper', {}).get('year', 0)
        target_year = s.get('target_paper', {}).get('year', 0)
        if source_year > 0 and target_year > 0 and source_year < target_year:
            issues['time_violations'] += 1
    
    # 报告问题
    all_ok = True
    for issue, count in issues.items():
        if count > 0:
            print(f"  ⚠️  {issue}: {count} 个样本")
            all_ok = False
    
    if all_ok:
        print(f"  ✅ 采样检查通过")
    
    # 统计信息
    print(f"\n📈 数据统计:")
    
    # 统计年份分布
    years = []
    for s in samples_to_check:
        if s.get('source_paper', {}).get('year'):
            years.append(s['source_paper']['year'])
    
    if years:
        print(f"  源论文年份范围: {min(years)} - {max(years)}")
    
    # 统计类别
    categories = defaultdict(int)
    for s in samples_to_check:
        cats = s.get('source_paper', {}).get('categories', [])
        if isinstance(cats, list):
            for cat in cats:
                categories[cat] += 1
    
    if categories:
        print(f"  主要类别 (前5):")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {cat}: {count}")
    
    # 检查citation_context长度
    context_lengths = [len(s.get('citation_context', '').split()) for s in samples_to_check]
    if context_lengths:
        avg_length = sum(context_lengths) / len(context_lengths)
        print(f"  平均citation_context长度: {avg_length:.1f} 单词")
        print(f"  最短: {min(context_lengths)}, 最长: {max(context_lengths)}")
    
    return True

def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    
    files_to_check = [
        ("train.json", "训练集"),
        ("val.json", "验证集"),
        ("test.json", "测试集")
    ]
    
    all_ok = True
    for filename, dataset_name in files_to_check:
        file_path = data_dir / filename
        if file_path.exists():
            try:
                check_data_file(str(file_path), dataset_name)
            except Exception as e:
                print(f"\n❌ 检查 {filename} 时出错: {e}")
                all_ok = False
        else:
            print(f"\n⚠️  文件不存在: {file_path}")
    
    print(f"\n{'='*60}")
    if all_ok:
        print("✅ 数据格式检查完成")
    else:
        print("⚠️  发现一些问题，请查看上面的报告")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

