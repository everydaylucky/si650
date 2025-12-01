"""
实验管理系统
统一管理所有实验的运行、结果存储和对比分析
"""
import os
import json
import yaml
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd

@dataclass
class ExperimentResult:
    """实验结果数据类"""
    experiment_id: str
    experiment_name: str
    model_type: str
    variant: str  # zero-shot, fine-tuned, etc.
    timestamp: str
    config: Dict
    metrics: Dict[str, float]
    training_info: Optional[Dict] = None
    model_path: Optional[str] = None
    notes: Optional[str] = None

class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.results_dir / "all_experiments.json"
        self.summary_file = self.results_dir / "experiment_summary.csv"
        self.experiments: List[ExperimentResult] = []
        self._load_experiments()
    
    def _load_experiments(self):
        """加载已有实验"""
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file, 'r') as f:
                    data = json.load(f)
                    self.experiments = [ExperimentResult(**exp) for exp in data]
            except Exception as e:
                print(f"⚠ 加载实验记录失败: {e}")
                self.experiments = []
    
    def _save_experiments(self):
        """保存实验记录"""
        try:
            data = [asdict(exp) for exp in self.experiments]
            with open(self.experiments_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # 更新CSV摘要
            self._update_summary()
        except Exception as e:
            print(f"❌ 保存实验记录失败: {e}")
            traceback.print_exc()
    
    def _update_summary(self):
        """更新实验摘要CSV"""
        if not self.experiments:
            return
        
        rows = []
        for exp in self.experiments:
            row = {
                'experiment_id': exp.experiment_id,
                'experiment_name': exp.experiment_name,
                'model_type': exp.model_type,
                'variant': exp.variant,
                'timestamp': exp.timestamp,
                'mrr': exp.metrics.get('mrr', 0),
                'recall@5': exp.metrics.get('recall@5', 0),
                'recall@10': exp.metrics.get('recall@10', 0),
                'recall@20': exp.metrics.get('recall@20', 0),
                'ndcg@10': exp.metrics.get('ndcg@10', 0),
                'ndcg@20': exp.metrics.get('ndcg@20', 0),
                'model_path': exp.model_path or '',
                'notes': exp.notes or ''
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.summary_file, index=False, encoding='utf-8')
    
    def save_experiment(self,
                       experiment_name: str,
                       model_type: str,
                       variant: str,
                       metrics: Dict[str, float],
                       config: Dict,
                       training_info: Optional[Dict] = None,
                       model_path: Optional[str] = None,
                       notes: Optional[str] = None) -> str:
        """保存实验结果"""
        experiment_id = f"{model_type}_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            model_type=model_type,
            variant=variant,
            timestamp=datetime.now().isoformat(),
            config=config,
            metrics=metrics,
            training_info=training_info,
            model_path=model_path,
            notes=notes
        )
        
        # 保存详细结果
        result_file = self.results_dir / f"{experiment_id}.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        
        # 添加到列表
        self.experiments.append(result)
        self._save_experiments()
        
        print(f"\n✓ 实验结果已保存:")
        print(f"  ID: {experiment_id}")
        print(f"  文件: {result_file}")
        
        return experiment_id
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """获取实验"""
        for exp in self.experiments:
            if exp.experiment_id == experiment_id:
                return exp
        return None
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """对比多个实验"""
        exps = [self.get_experiment(eid) for eid in experiment_ids]
        exps = [e for e in exps if e is not None]
        
        if not exps:
            return pd.DataFrame()
        
        rows = []
        for exp in exps:
            row = {
                'experiment_name': exp.experiment_name,
                'model_type': exp.model_type,
                'variant': exp.variant,
                **exp.metrics
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_best_experiment(self, metric: str = 'mrr') -> Optional[ExperimentResult]:
        """获取最佳实验"""
        if not self.experiments:
            return None
        
        best = max(self.experiments, key=lambda x: x.metrics.get(metric, 0))
        return best
    
    def list_experiments(self, model_type: Optional[str] = None) -> List[ExperimentResult]:
        """列出所有实验"""
        if model_type:
            return [e for e in self.experiments if e.model_type == model_type]
        return self.experiments

