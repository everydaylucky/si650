#!/usr/bin/env python3
"""
运行实验脚本
"""
import sys
import os
import yaml
import json
import traceback
import argparse
from pathlib import Path
from tqdm import tqdm

# 配置Hugging Face镜像（中国用户）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
print(f"使用Hugging Face镜像: {os.environ.get('HF_ENDPOINT')}")

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import MultiStagePipeline
from src.evaluation import Evaluator
from src.utils import load_json

def main():
    parser = argparse.ArgumentParser(description='运行多阶段检索实验')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径 (默认: config/model_config.yaml)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录路径 (默认: data/processed)')
    parser.add_argument('--data_file', type=str, default='test.json',
                        help='数据文件名 (默认: test.json)')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        print("=" * 60)
        print("开始加载配置...")
        if args.config:
            config_path = Path(args.config)
            if not config_path.is_absolute():
                config_path = project_root / config_path
        else:
            config_path = project_root / "config" / "model_config.yaml"
        
        if not config_path.exists():
            print(f"❌ 错误: 配置文件不存在: {config_path}")
            return
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"✓ 配置加载成功: {config_path}")
        
        # 加载数据
        print("\n" + "=" * 60)
        print("开始加载数据...")
        if args.data_dir:
            data_dir = Path(args.data_dir)
            if not data_dir.is_absolute():
                data_dir = project_root / data_dir
        else:
            data_dir = project_root / "data" / "processed"
        
        data_path = data_dir / args.data_file
        if not data_path.exists():
            print(f"❌ 错误: 数据文件不存在: {data_path}")
            print("请先准备测试数据")
            return
        
        test_data = load_json(str(data_path))
        print(f"✓ 数据加载成功: {len(test_data)} 个样本 (来源: {data_path})")
        
        # 初始化管道
        print("\n" + "=" * 60)
        print("初始化多阶段管道...")
        try:
            pipeline = MultiStagePipeline(config)
            print("✓ 管道初始化成功")
        except Exception as e:
            print(f"❌ 管道初始化失败:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            print(f"   详细堆栈:")
            traceback.print_exc()
            return
        
        # 加载文档（需要从数据中提取）
        print("\n" + "=" * 60)
        print("提取文档信息...")
        documents = []
        seen_ids = set()
        for item in test_data:
            # 添加 target_paper
            if "target_paper" in item:
                target = item["target_paper"]
                target_id = target.get("id") or target.get("paper_id")
                if target_id and target_id not in seen_ids:
                    documents.append(target)
                    seen_ids.add(target_id)
            
            # 添加 source_paper（用于查询增强）
            if "source_paper" in item:
                source = item["source_paper"]
                source_id = source.get("id") or source.get("paper_id")
                if source_id and source_id not in seen_ids:
                    documents.append(source)
                    seen_ids.add(source_id)
        
        if documents:
            print(f"构建索引 ({len(documents)} 个文档)...")
            try:
                pipeline.build_indices(documents)
                print("✓ 索引构建成功")
            except Exception as e:
                print(f"❌ 索引构建失败:")
                print(f"   错误类型: {type(e).__name__}")
                print(f"   错误信息: {str(e)}")
                print(f"   详细堆栈:")
                traceback.print_exc()
                return
        else:
            print("⚠ 警告: 未找到文档信息，跳过索引构建")
        
        # 运行实验
        print("\n" + "=" * 60)
        print("开始运行实验...")
        predictions = []
        ground_truth = []
        error_count = 0
        
        for i, query_data in enumerate(tqdm(test_data, desc="处理查询", unit="个")):
            try:
                # 提取 citation_context 和前后文
                citation_context = query_data.get("citation_context", "")
                context_before = ""
                context_after = ""
                
                # 如果 citation_context 是字典，提取前后文
                if isinstance(citation_context, dict):
                    context_before = citation_context.get("context_before", "")
                    context_after = citation_context.get("context_after", "")
                    citation_context = citation_context.get("text", "")
                
                query = {
                    "citation_context": citation_context,
                    "context_before": context_before,  # 添加前后文
                    "context_after": context_after,
                    "source_paper_id": query_data.get("source_paper_id", ""),
                    "source_paper": query_data.get("source_paper", {}),
                    "source_categories": query_data.get("source_categories", []),
                    "source_year": query_data.get("source_year", 2024)
                }
                
                results = pipeline.retrieve(query)
                predictions.append(results)
                
                # 提取ground truth
                if "target_paper_id" in query_data:
                    ground_truth.append(query_data["target_paper_id"])
                elif "target_paper" in query_data and "id" in query_data["target_paper"]:
                    ground_truth.append(query_data["target_paper"]["id"])
                else:
                    ground_truth.append(None)
            except Exception as e:
                error_count += 1
                print(f"\n❌ 处理第 {i+1} 个查询时出错:")
                print(f"   错误类型: {type(e).__name__}")
                print(f"   错误信息: {str(e)}")
                print(f"   查询数据: {json.dumps(query_data, indent=2, ensure_ascii=False)[:200]}...")
                print(f"   详细堆栈:")
                traceback.print_exc()
                predictions.append([])
                ground_truth.append(None)
        
        if error_count > 0:
            print(f"\n⚠ 警告: 共 {error_count} 个查询处理失败")
        
        # 评估
        print("\n" + "=" * 60)
        print("开始评估结果...")
        try:
            evaluator = Evaluator()
            metrics = evaluator.evaluate(predictions, ground_truth)
            print("✓ 评估完成")
        except Exception as e:
            print(f"❌ 评估失败:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            print(f"   详细堆栈:")
            traceback.print_exc()
            return
        
        # 保存结果
        results_dir = project_root / "experiments" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = results_dir / "experiment_results.json"
        try:
            with open(results_path, 'w') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"✓ 结果已保存到: {results_path}")
        except Exception as e:
            print(f"❌ 保存结果失败:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("实验结果:")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 程序执行失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        print(f"   详细堆栈:")
        traceback.print_exc()

if __name__ == "__main__":
    main()

