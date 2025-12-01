#!/usr/bin/env python3
"""
实验结果分析和对比
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.experiment_manager import ExperimentManager

def main():
    parser = argparse.ArgumentParser(description='分析实验结果')
    parser.add_argument('--compare', nargs='+', type=str, default=None,
                        help='对比指定实验ID')
    parser.add_argument('--model_type', type=str, default=None,
                        help='按模型类型筛选')
    parser.add_argument('--variant', type=str, default=None,
                        help='按variant筛选')
    parser.add_argument('--output', type=str, default='experiments/results/analysis_report.md',
                        help='输出报告路径')
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    print("=" * 80)
    print("实验结果分析")
    print("=" * 80)
    
    # 获取实验
    if args.compare:
        exps = [manager.get_experiment(eid) for eid in args.compare]
        exps = [e for e in exps if e is not None]
        title = "实验对比"
    else:
        exps = manager.list_experiments()
        if args.model_type:
            exps = [e for e in exps if e.model_type == args.model_type]
        if args.variant:
            exps = [e for e in exps if e.variant == args.variant]
        title = "所有实验"
    
    if not exps:
        print("❌ 没有找到实验")
        return
    
    # 创建对比表格
    df = manager.compare_experiments([e.experiment_id for e in exps])
    
    print(f"\n{title} ({len(exps)} 个实验):")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # 保存报告
    if args.output:
        output_path = project_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = f"""# 实验结果分析报告

## {title}

### 实验列表

{df.to_markdown(index=False)}

### 最佳实验

"""
        best = manager.get_best_experiment('mrr')
        if best:
            report += f"""
**最佳MRR**: {best.experiment_name}
- MRR: {best.metrics['mrr']:.4f}
- Recall@10: {best.metrics['recall@10']:.4f}
- NDCG@10: {best.metrics['ndcg@10']:.4f}
- 模型路径: {best.model_path or 'N/A'}

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ 报告已保存到: {output_path}")

if __name__ == "__main__":
    main()

