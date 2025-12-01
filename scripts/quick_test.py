#!/usr/bin/env python3
"""
快速测试脚本 - 验证系统是否正常工作
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有关键模块导入"""
    print("测试模块导入...")
    
    try:
        from src.experiments import ExperimentManager, ALL_EXPERIMENTS
        print("✓ 实验管理系统")
    except Exception as e:
        print(f"❌ 实验管理系统: {e}")
        return False
    
    try:
        from src.pipeline import MultiStagePipeline
        print("✓ 多阶段管道")
    except Exception as e:
        print(f"❌ 多阶段管道: {e}")
        return False
    
    try:
        from src.training.trainer import SciBERTTrainer
        print("✓ 训练器")
    except Exception as e:
        print(f"❌ 训练器: {e}")
        return False
    
    try:
        from src.models.retrieval import BM25Retriever, PRFRetriever
        print("✓ 检索模型")
    except Exception as e:
        print(f"❌ 检索模型: {e}")
        return False
    
    try:
        from src.evaluation import Evaluator
        print("✓ 评估器")
    except Exception as e:
        print(f"❌ 评估器: {e}")
        return False
    
    return True

def test_experiment_config():
    """测试实验配置"""
    print("\n测试实验配置...")
    
    try:
        from src.experiments.experiment_config import ALL_EXPERIMENTS
        print(f"✓ 已定义 {len(ALL_EXPERIMENTS)} 个实验")
        
        # 检查关键实验
        key_experiments = [
            'exp_1_1_bm25',
            'exp_2_1_scibert_zs',
            'exp_3_1_scibert_ft'
        ]
        
        for exp_id in key_experiments:
            if exp_id in ALL_EXPERIMENTS:
                print(f"  ✓ {exp_id}")
            else:
                print(f"  ❌ {exp_id} 缺失")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 实验配置: {e}")
        return False

def test_data_files():
    """测试数据文件"""
    print("\n测试数据文件...")
    
    data_dir = project_root / "data" / "processed" / "fast_experiment"
    required_files = ["train.json", "val.json", "test.json"]
    
    all_exist = True
    for file in required_files:
        path = data_dir / file
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ❌ {file} 不存在")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 60)
    print("系统快速测试")
    print("=" * 60)
    
    results = []
    
    results.append(("模块导入", test_imports()))
    results.append(("实验配置", test_experiment_config()))
    results.append(("数据文件", test_data_files()))
    
    print("\n" + "=" * 60)
    print("测试结果:")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"{name:20s}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！系统就绪。")
    else:
        print("\n⚠ 部分测试失败，请检查上述错误。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

