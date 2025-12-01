#!/usr/bin/env python3
"""
完整实验流程：训练 + 评估
"""
import sys
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description='运行完整实验流程（训练+评估）')
    parser.add_argument('--config', type=str, default='config/fast_experiment_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='data/processed/fast_experiment',
                        help='数据目录路径')
    parser.add_argument('--skip_training', action='store_true',
                        help='跳过训练，直接评估')
    parser.add_argument('--train_only', action='store_true',
                        help='只训练，不评估')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("完整实验流程")
    print("=" * 60)
    
    # 步骤1: 训练模型
    if not args.skip_training:
        print("\n" + "=" * 60)
        print("步骤1: 训练SciBERT模型")
        print("=" * 60)
        
        train_cmd = [
            sys.executable,
            str(project_root / "scripts" / "train_scibert.py"),
            "--config", args.config
        ]
        
        try:
            result = subprocess.run(train_cmd, check=True, cwd=str(project_root))
            print("\n✓ 训练完成")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 训练失败: {e}")
            return
        except KeyboardInterrupt:
            print("\n⚠ 训练被中断")
            return
    else:
        print("\n⚠ 跳过训练步骤")
    
    # 步骤2: 运行评估
    if not args.train_only:
        print("\n" + "=" * 60)
        print("步骤2: 运行评估")
        print("=" * 60)
        
        # 检查是否有训练后的配置
        config_path = project_root / args.config
        trained_config_path = project_root / "config" / "fast_experiment_config_trained.yaml"
        
        eval_config = str(trained_config_path) if trained_config_path.exists() else args.config
        
        eval_cmd = [
            sys.executable,
            str(project_root / "scripts" / "run_experiment.py"),
            "--config", eval_config,
            "--data_dir", args.data_dir
        ]
        
        try:
            result = subprocess.run(eval_cmd, check=True, cwd=str(project_root))
            print("\n✓ 评估完成")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 评估失败: {e}")
            return
        except KeyboardInterrupt:
            print("\n⚠ 评估被中断")
            return
    
    print("\n" + "=" * 60)
    print("✓ 完整实验流程完成！")
    print("=" * 60)
    print(f"\n结果文件:")
    print(f"  - 模型: experiments/checkpoints/scibert/")
    print(f"  - 评估结果: experiments/results/experiment_results.json")

if __name__ == "__main__":
    main()

