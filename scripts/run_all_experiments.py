#!/usr/bin/env python3
"""
运行所有实验的主脚本
支持运行单个实验、整个track、或所有实验
"""
import sys
import argparse
import traceback
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.experiment_manager import ExperimentManager
from src.experiments.experiment_config import ALL_EXPERIMENTS, get_experiment_config
from src.pipeline import MultiStagePipeline
from src.evaluation import Evaluator
from src.utils import load_json
import yaml

def run_single_experiment(exp_id: str, data_dir: str, manager: ExperimentManager, fast_mode: bool = False, sample_size: int = None, sample_ratio: float = None, random_seed: int = 42):
    """运行单个实验"""
    config = get_experiment_config(exp_id)
    if not config:
        print(f"❌ 未知实验ID: {exp_id}")
        return False
    
    print("=" * 80)
    print(f"运行实验: {config.name}")
    print(f"模型类型: {config.model_type}")
    print(f"变体: {config.variant}")
    print(f"需要训练: {config.requires_training}")
    print("=" * 80)
    
    try:
        # 加载配置
        config_path = project_root / config.config_path
        if config_path.exists():
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
        else:
            print(f"⚠ 配置文件不存在: {config_path}，使用默认配置")
            config_dict = create_default_config(config)
        
        # 检查数据目录格式
        data_path = Path(data_dir)
        corpus_file = data_path / "corpus.json"
        test_file = data_path / "test.json"
        
        # 如果存在 corpus.json，说明是新格式（分离的索引和测试数据）
        use_separated_format = corpus_file.exists() and test_file.exists()
        
        if use_separated_format:
            print(f"\n✓ 检测到新格式数据（分离的索引和测试数据）")
            print(f"  索引文档: {corpus_file}")
            print(f"  测试数据: {test_file}")
        else:
            print(f"\n使用传统格式数据: {data_dir}")
        
        # 训练模型（如果需要）
        model_path = None
        training_info = None
        if config.requires_training:
            print(f"\n需要训练模型: {config.model_type}")
            model_path = train_model(config, config_dict, data_dir)
            if not model_path:
                print(f"❌ 训练失败，跳过实验")
                return False
            training_info = {"model_path": model_path, "status": "completed"}
            
            # 更新配置中的模型路径
            if config.model_type == "l2r":
                model_file_path = Path(model_path)
                if model_file_path.exists():
                    abs_path = str(model_file_path.absolute())
                    config_dict.setdefault("stage3", {}).setdefault("l2r", {})["model_path"] = abs_path
                    print(f"✓ 已更新配置中的L2R模型路径: {abs_path}")
                else:
                    print(f"⚠ 模型文件不存在: {model_path}")
            elif config.model_type == "scibert":
                if Path(model_path).exists():
                    config_dict.setdefault("stage2", {}).setdefault("bi_encoder", {})["fine_tuned_path"] = model_path
            elif config.model_type == "cross_encoder":
                if Path(model_path).exists():
                    config_dict.setdefault("stage3", {}).setdefault("cross_encoder", {})["fine_tuned_path"] = model_path
            elif config.model_type == "specter2":
                if Path(model_path).exists():
                    config_dict.setdefault("stage1", {}).setdefault("specter2", {})["fine_tuned_path"] = model_path
        else:
            training_info = {"status": "not_required"}
        
        # 自动修复配置
        model_type = config.model_type
        stage1_config = config_dict.setdefault("stage1", {})
        stage2_config = config_dict.setdefault("stage2", {})
        stage3_config = config_dict.setdefault("stage3", {})
        
        if model_type == "pipeline":
            # Pipeline类型需要确保其内部依赖的检索器被启用
            if not stage1_config.get("use_bm25", False):
                stage1_config["use_bm25"] = True
                print("⚠ 检测到Pipeline实验需要Stage1检索器，自动启用BM25...")
            if not stage1_config.get("use_specter2", False):
                stage1_config["use_specter2"] = True
                print("⚠ 检测到Pipeline实验需要Stage1检索器，自动启用SPECTER2...")
        elif model_type == "specter2":
            # 确保SPECTER2实验启用了SPECTER2检索器
            if not stage1_config.get("use_specter2", False):
                print(f"⚠ 检测到SPECTER2实验需要Stage1 SPECTER2，自动启用...")
                stage1_config["use_specter2"] = True
                stage1_config["use_bm25"] = False
                stage1_config["use_tfidf"] = False
                stage1_config["use_prf"] = False
        
        # 评估实验
        print("\n" + "=" * 80)
        print("开始评估...")
        print("=" * 80)
        results = evaluate_experiment(config, config_dict, data_dir, model_path, use_separated_format, fast_mode, sample_size, sample_ratio, random_seed)
        
        if not results:
            print("❌ 评估失败")
            return False
        
        # 保存结果
        result_id = manager.save_experiment(
            experiment_name=results.get("experiment_name", config.name),
            model_type=results.get("model_type", config.model_type),
            variant=results.get("variant", config.variant),
            metrics=results.get("metrics", {}),
            config=results.get("config", config_dict),
            training_info=results.get("training_info", training_info),
            model_path=results.get("model_path", model_path),
            notes=results.get("notes", config.description if hasattr(config, 'description') else "")
        )
        print(f"\n✓ 结果已保存: {result_id}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 实验失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        traceback.print_exc()
        return False

def train_model(config, config_dict: dict, data_dir: str) -> str:
    """训练模型"""
    import subprocess
    
    model_type = config.model_type
    variant = config.variant if hasattr(config, 'variant') else "zero-shot"
    
    if model_type == "l2r":
        train_file = Path(data_dir) / "train.json"
        val_file = Path(data_dir) / "val.json"
        output_dir = project_root / "experiments" / "checkpoints" / "l2r" / variant
        
        model_path = output_dir / "l2r_model.txt"
        if model_path.exists():
            print(f"✓ 模型已存在: {model_path}")
            return str(model_path)
        
        use_fine_tuned = (variant == "fine-tuned")
        
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "train_l2r.py"),
            "--train_file", str(train_file),
            "--val_file", str(val_file) if val_file.exists() else "",
            "--output_dir", str(output_dir),
            "--variant", variant,
            "--num_leaves", "31",
            "--learning_rate", "0.05",
            "--n_estimators", "300",
            "--max_depth", "6"
        ]
        if use_fine_tuned:
            cmd.append("--use_fine_tuned")
        
        print(f"运行训练命令...")
        print("=" * 80)
        result = subprocess.run(cmd, cwd=str(project_root))
        
        print("=" * 80)
        if result.returncode != 0:
            print(f"❌ 训练失败 (退出码: {result.returncode})")
            return None
        
        if model_path.exists():
            print(f"✓ 模型已保存: {model_path}")
            return str(model_path)
        else:
            print(f"⚠ 模型文件未找到: {model_path}")
            return None
    
    elif model_type == "scibert":
        import subprocess
        train_file = Path(data_dir) / "train.json"
        val_file = Path(data_dir) / "val.json"
        output_dir = project_root / "experiments" / "checkpoints" / "scibert"
        
        model_files = [
            output_dir / "pytorch_model.bin",
            output_dir / "model.safetensors",
            output_dir / "config.json",
            output_dir / "modules.json"
        ]
        model_exists = any(f.exists() for f in model_files)
        
        if model_exists:
            print(f"✓ 模型已存在: {output_dir}")
            return str(output_dir)
        
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "train_scibert.py"),
            "--train_file", str(train_file),
            "--val_file", str(val_file) if val_file.exists() else "",
            "--output_dir", str(output_dir),
            "--epochs", "3",
            "--batch_size", "16",
            "--learning_rate", "2e-5"
        ]
        
        print(f"运行训练命令...")
        print("=" * 80)
        result = subprocess.run(cmd, cwd=str(project_root))
        
        print("=" * 80)
        if result.returncode != 0:
            print(f"❌ 训练失败 (退出码: {result.returncode})")
            return None
        
        model_exists = any(f.exists() for f in model_files)
        if model_exists:
            print(f"✓ 模型已保存: {output_dir}")
            return str(output_dir)
        else:
            print(f"⚠ 模型文件未找到: {output_dir}")
            return None
    
    elif model_type == "cross_encoder":
        import subprocess
        train_file = Path(data_dir) / "train.json"
        val_file = Path(data_dir) / "val.json"
        output_dir = project_root / "experiments" / "checkpoints" / "cross_encoder"
        
        model_files = [
            output_dir / "pytorch_model.bin",
            output_dir / "model.safetensors",
            output_dir / "config.json"
        ]
        model_exists = any(f.exists() for f in model_files)
        
        if model_exists:
            print(f"✓ 模型已存在: {output_dir}")
            return str(output_dir)
        
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "train_cross_encoder.py"),
            "--train_file", str(train_file),
            "--val_file", str(val_file) if val_file.exists() else "",
            "--output_dir", str(output_dir),
            "--epochs", "3",
            "--batch_size", "32",
            "--learning_rate", "2e-5"
        ]
        
        print(f"运行训练命令...")
        print("=" * 80)
        result = subprocess.run(cmd, cwd=str(project_root))
        
        print("=" * 80)
        if result.returncode != 0:
            print(f"❌ 训练失败 (退出码: {result.returncode})")
            return None
        
        model_exists = any(f.exists() for f in model_files)
        if model_exists:
            print(f"✓ 模型已保存: {output_dir}")
            return str(output_dir)
        else:
            print(f"⚠ 模型文件未找到: {output_dir}")
            return None
    
    elif model_type == "specter2":
        import subprocess
        variant = config.variant if hasattr(config, 'variant') else "zero-shot"
        
        train_file = Path(data_dir) / "train.json"
        val_file = Path(data_dir) / "val.json"
        output_dir = project_root / "experiments" / "checkpoints" / "specter2"
        
        model_files = [
            output_dir / "pytorch_model.bin",
            output_dir / "model.safetensors",
            output_dir / "config.json",
            output_dir / "modules.json"
        ]
        model_exists = any(f.exists() for f in model_files)
        
        if model_exists:
            print(f"✓ 模型已存在: {output_dir}")
            return str(output_dir)
        
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "train_specter2.py"),
            "--train_file", str(train_file),
            "--val_file", str(val_file) if val_file.exists() else "",
            "--output_dir", str(output_dir),
            "--epochs", "3",
            "--batch_size", "16",
            "--learning_rate", "2e-5"
        ]
        
        print(f"运行训练命令...")
        print("=" * 80)
        result = subprocess.run(cmd, cwd=str(project_root))
        
        print("=" * 80)
        if result.returncode != 0:
            print(f"❌ 训练失败 (退出码: {result.returncode})")
            return None
        
        model_exists = any(f.exists() for f in model_files)
        if model_exists:
            print(f"✓ 模型已保存: {output_dir}")
            return str(output_dir)
        else:
            print(f"⚠ 模型文件未找到: {output_dir}")
            if output_dir.exists():
                print(f"   目录内容: {list(output_dir.iterdir())}")
            return None
    
    return None

def evaluate_experiment(config, config_dict: dict, data_dir: str, model_path: str = None, use_separated_format: bool = False, fast_mode: bool = False, sample_size: int = None, sample_ratio: float = None, random_seed: int = 42) -> dict:
    """评估实验"""
    # 加载测试数据
    data_path = Path(data_dir)
    
    if use_separated_format:
        # 新格式：分离的索引和测试数据
        corpus_file = data_path / "corpus.json"
        test_file = data_path / "test.json"
        
        if not corpus_file.exists():
            print(f"❌ 索引文档文件不存在: {corpus_file}")
            return None
        if not test_file.exists():
            print(f"❌ 测试文件不存在: {test_file}")
            return None
        
        # 加载索引文档
        print(f"加载索引文档: {corpus_file}")
        corpus_documents = load_json(str(corpus_file))
        print(f"✓ 加载完成: {len(corpus_documents)} 个文档")
        
        # 加载测试数据
        print(f"加载测试数据: {test_file}")
        test_data = load_json(str(test_file))
        print(f"✓ 加载完成: {len(test_data)} 个样本")
        
        # Fast 模式：采样测试数据
        original_size = len(test_data)
        if fast_mode or sample_size or sample_ratio:
            import random
            random.seed(random_seed)
            
            if sample_size:
                n = min(sample_size, len(test_data))
            elif sample_ratio:
                n = int(len(test_data) * sample_ratio)
            elif fast_mode:
                n = min(472, len(test_data))  # 默认 472 个样本
            
            if n < len(test_data):
                test_data = random.sample(test_data, n)
                print(f"⚡ Fast 模式已启用")
                print(f"   采样: {n} / {original_size} 个样本 ({n/original_size*100:.1f}%)")
                print(f"   随机种子: {random_seed}")
            else:
                fast_mode = False
        
    else:
        # 传统格式：从测试数据中提取文档
        test_file = data_path / "test.json"
        if not test_file.exists():
            print(f"❌ 测试文件不存在: {test_file}")
            return None
        
        raw_data = load_json(str(test_file))
        
        # 处理不同的数据格式
        if isinstance(raw_data, dict) and 'samples' in raw_data:
            test_data = raw_data['samples']
            print(f"✓ 加载数据: {len(test_data)} 个样本 (字典格式，包含samples)")
        elif isinstance(raw_data, list):
            test_data = raw_data
            print(f"✓ 加载数据: {len(test_data)} 个样本 (列表格式)")
        else:
            print(f"❌ 未知数据格式: {type(raw_data)}")
            return None
        
        # Fast 模式：采样测试数据
        original_size = len(test_data)
        if fast_mode or sample_size or sample_ratio:
            import random
            random.seed(random_seed)
            
            if sample_size:
                n = min(sample_size, len(test_data))
            elif sample_ratio:
                n = int(len(test_data) * sample_ratio)
            elif fast_mode:
                n = min(472, len(test_data))  # 默认 472 个样本
            
            if n < len(test_data):
                test_data = random.sample(test_data, n)
                print(f"⚡ Fast 模式已启用")
                print(f"   采样: {n} / {original_size} 个样本 ({n/original_size*100:.1f}%)")
                print(f"   随机种子: {random_seed}")
            else:
                fast_mode = False
        
        # 从测试数据中提取文档（用于索引）
        corpus_documents = []
        seen_ids = set()
        for item in test_data:
            # 处理不同的数据格式
            if "candidates" in item:
                candidates = item.get("candidates", [])
                for candidate in candidates:
                    doc_id = candidate.get("arxiv_id", "")
                    if doc_id and doc_id not in seen_ids:
                        doc = {
                            "id": doc_id,
                            "paper_id": doc_id,
                            "title": candidate.get("title", ""),
                            "abstract": candidate.get("abstract", ""),
                            "categories": candidate.get("categories", "").split() if isinstance(candidate.get("categories"), str) else candidate.get("categories", []),
                            "year": candidate.get("year", 2020)
                        }
                        corpus_documents.append(doc)
                        seen_ids.add(doc_id)
            
            if "target_paper" in item:
                target = item["target_paper"]
                target_id = target.get("id") or target.get("paper_id")
                if target_id and target_id not in seen_ids:
                    corpus_documents.append(target)
                    seen_ids.add(target_id)
            
            if "negatives" in item:
                negatives = item.get("negatives", [])
                for neg in negatives:
                    if isinstance(neg, dict):
                        neg_id = neg.get("id") or neg.get("paper_id")
                        if neg_id and neg_id not in seen_ids:
                            corpus_documents.append(neg)
                            seen_ids.add(neg_id)
            
            # 添加 source_paper
            source_paper = item.get("source_paper", {})
            if source_paper:
                if "arxiv_id" in source_paper:
                    source = {
                        "id": source_paper.get("arxiv_id", ""),
                        "paper_id": source_paper.get("arxiv_id", ""),
                        "title": source_paper.get("title", ""),
                        "abstract": source_paper.get("abstract", ""),
                        "categories": source_paper.get("categories", "").split() if isinstance(source_paper.get("categories"), str) else source_paper.get("categories", []),
                        "year": source_paper.get("year", 2020)
                    }
                else:
                    source = source_paper
                
                source_id = source.get("id") or source.get("paper_id")
                if source_id and source_id not in seen_ids:
                    corpus_documents.append(source)
                    seen_ids.add(source_id)
    
    # 初始化管道
    pipeline = MultiStagePipeline(config_dict)
    
    # 构建索引
    if corpus_documents:
        print(f"\n构建索引 ({len(corpus_documents)} 个文档)...")
        pipeline.build_indices(corpus_documents)
    
    # 运行评估
    predictions = []
    ground_truth = []
    
    from tqdm import tqdm
    for query_data in tqdm(test_data, desc="评估中"):
        # 提取 citation_context 和前后文
        citation_context = query_data.get("citation_context", "")
        context_before = ""
        context_after = ""
        
        # 如果 citation_context 是字典，提取前后文（data/full 格式）
        if isinstance(citation_context, dict):
            context_before = citation_context.get("context_before", "")
            context_after = citation_context.get("context_after", "")
            citation_context = citation_context.get("text", "")
        
        # 提取 source_paper_id 和 source_paper
        source_paper = query_data.get("source_paper", {})
        source_paper_id = ""
        source_categories = []
        source_year = 2024
        
        if source_paper:
            if "arxiv_id" in source_paper:
                source_paper_id = source_paper.get("arxiv_id", "")
                source_categories = source_paper.get("categories", "").split() if isinstance(source_paper.get("categories"), str) else source_paper.get("categories", [])
                source_year = source_paper.get("year", 2024)
                # 转换格式以匹配现有代码
                source_paper = {
                    "id": source_paper_id,
                    "paper_id": source_paper_id,
                    "title": source_paper.get("title", ""),
                    "abstract": source_paper.get("abstract", ""),
                    "categories": source_categories,
                    "year": source_year
                }
            else:
                source_paper_id = source_paper.get("id") or source_paper.get("paper_id", "")
                source_categories = source_paper.get("categories", [])
                source_year = source_paper.get("year", 2024)
        else:
            source_paper_id = query_data.get("source_paper_id", "")
            source_categories = query_data.get("source_categories", [])
            source_year = query_data.get("source_year", 2024)
        
        query = {
            "citation_context": citation_context,
            "context_before": context_before,
            "context_after": context_after,
            "source_paper_id": source_paper_id,
            "source_paper": source_paper,
            "source_categories": source_categories,
            "source_year": source_year
        }
        
        results = pipeline.retrieve(query)
        predictions.append(results)
        
        # 提取 ground truth (处理不同的数据格式)
        target_paper_id = None
        
        # 格式1: 新格式 (target_paper_id 字段)
        if "target_paper_id" in query_data:
            target_paper_id = query_data["target_paper_id"]
        
        # 格式2: data/full 格式 (从 candidates 中找到 label=1 的)
        if not target_paper_id and "candidates" in query_data:
            candidates = query_data.get("candidates", [])
            for candidate in candidates:
                if candidate.get("label", 0) == 1:
                    target_paper_id = candidate.get("arxiv_id", "")
                    break
        
        # 格式3: fast_experiment 格式
        if not target_paper_id:
            if "target_paper_id" in query_data:
                target_paper_id = query_data["target_paper_id"]
            elif "target_paper" in query_data:
                target_paper = query_data["target_paper"]
                target_paper_id = target_paper.get("id") or target_paper.get("paper_id", "")
        
        ground_truth.append(target_paper_id)
    
    # 计算指标
    evaluator = Evaluator()
    metrics = evaluator.evaluate(predictions, ground_truth)
    
    # 记录 fast 模式信息
    fast_mode_info = None
    if 'fast_mode' in locals() and (fast_mode or sample_size or sample_ratio):
        fast_mode_info = {
            "enabled": True,
            "sample_size": len(test_data) if 'test_data' in locals() else None,
            "original_size": original_size if 'original_size' in locals() else None,
            "sample_ratio": len(test_data) / original_size if 'original_size' in locals() and original_size > 0 else None,
            "random_seed": random_seed
        }
    
    return {
        "experiment_id": config.name.lower().replace(" ", "_"),
        "experiment_name": config.name,
        "model_type": config.model_type,
        "variant": config.variant,
        "metrics": metrics,
        "config": config_dict,
        "training_info": training_info if 'training_info' in locals() else None,
        "model_path": model_path,
        "fast_mode": fast_mode_info,
        "notes": config.description if hasattr(config, 'description') else ""
    }

def create_default_config(exp_config) -> dict:
    """创建默认配置"""
    model_type = exp_config.model_type
    variant = exp_config.variant
    
    base_config = {
        "stage1": {
            "use_bm25": False,
            "use_tfidf": False,
            "use_specter2": False,
            "use_prf": False,
            "top_k": 1000
        },
        "stage2": {
            "use_bi_encoder": False,
            "use_colbert": False,
            "use_rrf": False,
            "top_k": 50
        },
        "stage3": {
            "use_cross_encoder": False,
            "use_l2r": False,
            "top_k": 20
        }
    }
    
    # 根据模型类型设置配置
    if model_type == "bm25":
        base_config["stage1"]["use_bm25"] = True
    elif model_type == "tfidf":
        base_config["stage1"]["use_tfidf"] = True
    elif model_type == "specter2":
        base_config["stage1"]["use_specter2"] = True
    elif model_type == "scibert":
        base_config["stage2"]["use_bi_encoder"] = True
        if variant == "fine-tuned":
            base_config["stage2"]["bi_encoder"] = {
                "fine_tuned_path": "experiments/checkpoints/scibert"
            }
    elif model_type == "colbert":
        base_config["stage2"]["use_colbert"] = True
    elif model_type == "cross_encoder":
        base_config["stage3"]["use_cross_encoder"] = True
        if variant == "fine-tuned":
            base_config["stage3"]["cross_encoder"] = {
                "fine_tuned_path": "experiments/checkpoints/cross_encoder"
            }
    elif model_type == "rrf":
        base_config["stage1"]["use_bm25"] = True
        base_config["stage1"]["use_specter2"] = True
        base_config["stage2"]["use_rrf"] = True
    elif model_type == "l2r":
        base_config["stage1"]["use_bm25"] = True
        base_config["stage1"]["use_specter2"] = True
        base_config["stage3"]["use_l2r"] = True
        if variant == "fine-tuned":
            base_config["stage3"]["l2r"] = {
                "model_path": "experiments/checkpoints/l2r/ft/l2r_model.txt"
            }
    elif model_type == "pipeline":
        base_config["stage1"]["use_bm25"] = True
        base_config["stage1"]["use_specter2"] = True
        base_config["stage2"]["use_bi_encoder"] = True
        base_config["stage2"]["use_rrf"] = True
        base_config["stage3"]["use_cross_encoder"] = True
        base_config["stage3"]["use_l2r"] = True
    
    # 对于需要Stage1检索器提供候选的实验，自动启用BM25和SPECTER2
    if model_type in ["scibert", "colbert", "cross_encoder", "rrf", "l2r", "pipeline"]:
        if not base_config["stage1"].get("use_bm25", False):
            base_config["stage1"]["use_bm25"] = True
            print("⚠ 检测到实验需要Stage1检索器，自动启用BM25...")
        if not base_config["stage1"].get("use_specter2", False):
            base_config["stage1"]["use_specter2"] = True
            print("⚠ 检测到实验需要Stage1检索器，自动启用SPECTER2...")
    
    return base_config

def main():
    parser = argparse.ArgumentParser(description='运行所有实验')
    parser.add_argument('--experiment', type=str, default=None,
                        help='运行单个实验ID (如: exp_1_1_bm25)')
    parser.add_argument('--track', type=int, default=None,
                        help='运行整个track (1-5)')
    parser.add_argument('--variant', type=str, default=None,
                        help='运行特定variant (zero-shot, fine-tuned)')
    parser.add_argument('--all', action='store_true',
                        help='运行所有实验')
    parser.add_argument('--data_dir', type=str, default='data/processed/fast_experiment',
                        help='数据目录')
    parser.add_argument('--skip_trained', action='store_true',
                        help='跳过已训练的实验')
    parser.add_argument('--fast', action='store_true',
                        help='Fast 模式：采样 472 个样本进行快速测试')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='指定采样数量（覆盖 --fast）')
    parser.add_argument('--sample_ratio', type=float, default=None,
                        help='指定采样比例，如 0.25 表示 25%%（覆盖 --fast）')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    print("=" * 80)
    print("实验管理系统")
    print("=" * 80)
    
    # 确定要运行的实验
    experiments_to_run = []
    
    if args.experiment:
        config = get_experiment_config(args.experiment)
        if config:
            experiments_to_run = [args.experiment]
        else:
            print(f"❌ 未知实验ID: {args.experiment}")
            return
    elif args.track:
        from src.experiments.experiment_config import list_experiments_by_track
        experiments_to_run = [exp.exp_id for exp in list_experiments_by_track(args.track)]
    elif args.variant:
        from src.experiments.experiment_config import list_experiments_by_variant
        experiments_to_run = [exp.exp_id for exp in list_experiments_by_variant(args.variant)]
    elif args.all:
        experiments_to_run = list(ALL_EXPERIMENTS.keys())
    else:
        print("请指定要运行的实验 (--experiment, --track, --variant, 或 --all)")
        return
    
    print(f"\n将运行 {len(experiments_to_run)} 个实验")
    print(f"数据目录: {args.data_dir}")
    
    # 运行实验
    success_count = 0
    fail_count = 0
    
    for exp_id in experiments_to_run:
        success = run_single_experiment(
            exp_id, 
            args.data_dir, 
            manager,
            fast_mode=args.fast,
            sample_size=args.sample_size,
            sample_ratio=args.sample_ratio,
            random_seed=args.random_seed
        )
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "=" * 80)
    print("实验完成")
    print("=" * 80)
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")

if __name__ == '__main__':
    main()
