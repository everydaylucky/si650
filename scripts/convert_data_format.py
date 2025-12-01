#!/usr/bin/env python3
"""
将现有数据格式转换为final_test要求的格式
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

def convert_sample(old_sample: Dict) -> Dict:
    """转换单个样本"""
    # 提取citation_context
    citation_context = old_sample.get('citation_context', {})
    if isinstance(citation_context, dict):
        context_text = citation_context.get('text', '')
        section = citation_context.get('section', 'Unknown')
    else:
        context_text = str(citation_context)
        section = 'Unknown'
    
    # 提取source_paper
    source_paper_old = old_sample.get('source_paper', {})
    source_paper = {
        "id": source_paper_old.get('arxiv_id', source_paper_old.get('id', '')),
        "title": source_paper_old.get('title', ''),
        "abstract": source_paper_old.get('abstract', ''),
        "categories": source_paper_old.get('categories', '').split() if isinstance(source_paper_old.get('categories'), str) else source_paper_old.get('categories', []),
        "year": source_paper_old.get('year', 2020)
    }
    
    # 提取candidates（包含positive和negatives）
    candidates = old_sample.get('candidates', [])
    
    # 找到positive（label=1 或 type='positive'）
    target_paper = None
    positive_idx = -1
    for i, candidate in enumerate(candidates):
        if candidate.get('label', 0) == 1 or candidate.get('type', '') == 'positive':
            target_paper = {
                "id": candidate.get('arxiv_id', candidate.get('id', '')),
                "title": candidate.get('title', ''),
                "abstract": candidate.get('abstract', ''),
                "categories": candidate.get('categories', '').split() if isinstance(candidate.get('categories'), str) else candidate.get('categories', []),
                "year": candidate.get('year', 2020)
            }
            positive_idx = i
            break
    
    # 如果没有找到positive，使用第一个candidate
    if target_paper is None and candidates:
        candidate = candidates[0]
        target_paper = {
            "id": candidate.get('arxiv_id', candidate.get('id', '')),
            "title": candidate.get('title', ''),
            "abstract": candidate.get('abstract', ''),
            "categories": candidate.get('categories', '').split() if isinstance(candidate.get('categories'), str) else candidate.get('categories', []),
            "year": candidate.get('year', 2020)
        }
        positive_idx = 0
    
    if target_paper is None:
        return None  # 跳过没有target的样本
    
    # 提取negatives（排除positive）
    negatives = []
    for i, candidate in enumerate(candidates):
        if i != positive_idx:  # 跳过positive
            negative = {
                "id": candidate.get('arxiv_id', candidate.get('id', '')),
                "title": candidate.get('title', ''),
                "abstract": candidate.get('abstract', ''),
                "categories": candidate.get('categories', '').split() if isinstance(candidate.get('categories'), str) else candidate.get('categories', []),
                "year": candidate.get('year', 2020)
            }
            negatives.append(negative)
    
    # 构建新格式的样本
    new_sample = {
        "citation_context": context_text,
        "source_paper_id": source_paper["id"],
        "target_paper_id": target_paper["id"],
        "source_paper": source_paper,
        "target_paper": target_paper,
        "metadata": {
            "section": section,
            "source_year": source_paper["year"],
            "target_year": target_paper["year"],
            "source_categories": source_paper["categories"]
        }
    }
    
    # 仅训练集添加negatives
    if negatives:
        new_sample["negatives"] = negatives
    
    return new_sample

def convert_file(input_file: str, output_file: str, dataset_type: str = "train"):
    """转换整个文件"""
    print(f"\n转换文件: {input_file} -> {output_file}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 提取samples
    if isinstance(raw_data, dict):
        samples = raw_data.get('samples', [])
        metadata = raw_data.get('metadata', {})
        print(f"原始数据: {len(samples)} 个样本")
        print(f"元数据: {metadata}")
    elif isinstance(raw_data, list):
        samples = raw_data
        print(f"原始数据: {len(samples)} 个样本（数组格式）")
    else:
        print(f"❌ 未知的数据格式")
        return False
    
    # 转换样本
    converted_samples = []
    skipped = 0
    
    for i, old_sample in enumerate(samples):
        new_sample = convert_sample(old_sample)
        if new_sample:
            converted_samples.append(new_sample)
        else:
            skipped += 1
        
        if (i + 1) % 1000 == 0:
            print(f"  已处理: {i+1}/{len(samples)}")
    
    print(f"\n转换完成:")
    print(f"  成功: {len(converted_samples)} 个样本")
    print(f"  跳过: {skipped} 个样本")
    
    # 保存新格式
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_samples, f, indent=2, ensure_ascii=False)
    
    print(f"  已保存到: {output_file}")
    return True

def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    
    files_to_convert = [
        ("train.json", "train"),
        ("val.json", "val"),
        ("test.json", "test")
    ]
    
    # 创建备份目录
    backup_dir = data_dir / "backup_original"
    backup_dir.mkdir(exist_ok=True)
    
    for filename, dataset_type in files_to_convert:
        input_file = data_dir / filename
        output_file = data_dir / f"{filename}.converted"
        
        if not input_file.exists():
            print(f"⚠️  文件不存在: {input_file}")
            continue
        
        # 备份原文件
        import shutil
        backup_file = backup_dir / filename
        if not backup_file.exists():
            print(f"\n备份原文件: {backup_file}")
            shutil.copy(input_file, backup_file)
        
        # 转换
        if convert_file(str(input_file), str(output_file), dataset_type):
            print(f"✅ {filename} 转换完成")
        else:
            print(f"❌ {filename} 转换失败")
    
    print(f"\n{'='*60}")
    print("转换完成！")
    print(f"\n新文件已保存为 *.converted")
    print("原文件已备份到 data/processed/backup_original/")
    print(f"\n如果转换结果正确，可以手动重命名：")
    print("  mv train.json.converted train.json")
    print("  mv val.json.converted val.json")
    print("  mv test.json.converted test.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

