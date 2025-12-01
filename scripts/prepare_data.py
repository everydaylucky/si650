#!/usr/bin/env python3
"""
数据准备脚本：将原始citation数据转换为final_test格式
"""
import json
import sys
from pathlib import Path

def convert_citation_data(input_file: str, output_file: str):
    """
    将citation ground truth数据转换为final_test格式
    
    Args:
        input_file: 输入的citation数据文件路径
        output_file: 输出的JSON文件路径
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted = []
    
    # 处理每个样本
    samples = data.get('samples', [])
    for sample in samples:
        source_paper = sample.get('source_paper', {})
        citation_context = sample.get('citation_context', {})
        target_papers = sample.get('target_papers', [])
        
        # 只处理有target papers的样本
        if not target_papers:
            continue
        
        # 为每个target paper创建一个样本
        for target in target_papers:
            # 解析categories
            source_categories = source_paper.get('categories', '').split()
            target_categories = target.get('categories', '').split()
            
            # 提取年份（如果有）
            source_year = int(source_paper.get('year', 2020))
            target_year = int(target.get('year', 2020))
            
            converted_sample = {
                "citation_context": citation_context.get('text', ''),
                "source_paper_id": source_paper.get('arxiv_id', ''),
                "target_paper_id": target.get('arxiv_id', ''),
                "source_paper": {
                    "id": source_paper.get('arxiv_id', ''),
                    "title": source_paper.get('title', ''),
                    "abstract": source_paper.get('abstract', ''),
                    "categories": source_categories,
                    "year": source_year
                },
                "target_paper": {
                    "id": target.get('arxiv_id', ''),
                    "title": target.get('title', ''),
                    "abstract": target.get('abstract', ''),
                    "categories": target_categories,
                    "year": target_year
                },
                "metadata": {
                    "section": citation_context.get('section', 'Unknown'),
                    "source_year": source_year,
                    "target_year": target_year,
                    "source_categories": source_categories
                }
            }
            converted.append(converted_sample)
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成: {len(converted)} 个样本")
    print(f"输出文件: {output_file}")

def check_data_quality(data_file: str):
    """检查数据质量"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n数据统计:")
    print(f"  总样本数: {len(data)}")
    
    # 检查必需字段
    missing_fields = []
    time_violations = 0
    
    for i, sample in enumerate(data):
        # 检查必需字段
        required = ['citation_context', 'source_paper_id', 'target_paper_id', 
                   'source_paper', 'target_paper']
        for field in required:
            if field not in sample:
                missing_fields.append(f"样本{i}: 缺少字段 {field}")
        
        # 检查时间一致性
        source_year = sample.get('source_paper', {}).get('year', 0)
        target_year = sample.get('target_paper', {}).get('year', 0)
        if source_year < target_year:
            time_violations += 1
    
    if missing_fields:
        print(f"\n警告: 发现 {len(missing_fields)} 个缺少字段的样本")
        for msg in missing_fields[:5]:  # 只显示前5个
            print(f"  {msg}")
    
    if time_violations > 0:
        print(f"\n警告: 发现 {time_violations} 个时间违规样本（源论文年份 < 目标论文年份）")
    
    print("\n数据质量检查完成")

def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python prepare_data.py convert <input_file> <output_file>")
        print("  python prepare_data.py check <data_file>")
        return
    
    command = sys.argv[1]
    
    if command == "convert":
        if len(sys.argv) < 4:
            print("错误: 需要输入文件和输出文件路径")
            return
        convert_citation_data(sys.argv[2], sys.argv[3])
    elif command == "check":
        if len(sys.argv) < 3:
            print("错误: 需要数据文件路径")
            return
        check_data_quality(sys.argv[2])
    else:
        print(f"未知命令: {command}")

if __name__ == "__main__":
    main()

