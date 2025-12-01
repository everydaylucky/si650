"""
实验配置定义
定义所有实验的配置和元数据
"""
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """单个实验配置"""
    name: str
    model_type: str
    variant: str
    stage: int
    config_path: str
    requires_training: bool
    training_time_estimate: str
    description: str

# 所有实验定义
ALL_EXPERIMENTS = {
    # Track 1: Traditional IR Baselines
    "exp_1_1_bm25": ExperimentConfig(
        name="BM25 Baseline",
        model_type="bm25",
        variant="baseline",
        stage=1,
        config_path="config/experiments/exp_1_1_bm25.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="Traditional BM25 sparse retrieval baseline"
    ),
    "exp_1_2_tfidf": ExperimentConfig(
        name="TF-IDF Baseline",
        model_type="tfidf",
        variant="baseline",
        stage=1,
        config_path="config/experiments/exp_1_2_tfidf.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="TF-IDF sparse retrieval baseline"
    ),
    "exp_1_3_prf": ExperimentConfig(
        name="Query Expansion + BM25",
        model_type="prf",
        variant="baseline",
        stage=1,
        config_path="config/experiments/exp_1_3_prf.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="Pseudo-relevance feedback with BM25"
    ),
    
    # Track 2: Zero-shot Dense Models
    "exp_2_1_scibert_zs": ExperimentConfig(
        name="SciBERT Zero-shot",
        model_type="scibert",
        variant="zero-shot",
        stage=2,
        config_path="config/experiments/exp_2_1_scibert_zs.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="SciBERT bi-encoder zero-shot"
    ),
    "exp_2_2_specter2_zs": ExperimentConfig(
        name="SPECTER2 Zero-shot",
        model_type="specter2",
        variant="zero-shot",
        stage=1,
        config_path="config/experiments/exp_2_2_specter2_zs.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="SPECTER2 dense retrieval zero-shot"
    ),
    "exp_2_3_colbert_zs": ExperimentConfig(
        name="ColBERT Zero-shot",
        model_type="colbert",
        variant="zero-shot",
        stage=2,
        config_path="config/experiments/exp_2_3_colbert_zs.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="ColBERT late interaction zero-shot"
    ),
    "exp_2_4_crossenc_zs": ExperimentConfig(
        name="Cross-Encoder Zero-shot",
        model_type="cross_encoder",
        variant="zero-shot",
        stage=3,
        config_path="config/experiments/exp_2_4_crossenc_zs.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="Cross-Encoder reranker zero-shot (MS-MARCO)"
    ),
    
    # Track 3: Fine-tuned Models
    "exp_3_1_scibert_ft": ExperimentConfig(
        name="SciBERT Fine-tuned",
        model_type="scibert",
        variant="fine-tuned",
        stage=2,
        config_path="config/experiments/exp_3_1_scibert_ft.yaml",
        requires_training=True,
        training_time_estimate="3-4h",
        description="SciBERT bi-encoder fine-tuned on citation data"
    ),
    "exp_3_2_specter2_ft": ExperimentConfig(
        name="SPECTER2 Fine-tuned",
        model_type="specter2",
        variant="fine-tuned",
        stage=1,
        config_path="config/experiments/exp_3_2_specter2_ft.yaml",
        requires_training=True,
        training_time_estimate="4-5h",
        description="SPECTER2 dense retrieval fine-tuned"
    ),
    "exp_3_3_crossenc_ft": ExperimentConfig(
        name="Cross-Encoder Fine-tuned",
        model_type="cross_encoder",
        variant="fine-tuned",
        stage=3,
        config_path="config/experiments/exp_3_3_crossenc_ft.yaml",
        requires_training=True,
        training_time_estimate="5-6h",
        description="Cross-Encoder reranker fine-tuned"
    ),
    
    # Track 4: Fusion Methods
    "exp_4_1_rrf_zs": ExperimentConfig(
        name="RRF (Zero-shot)",
        model_type="rrf",
        variant="zero-shot",
        stage=2,
        config_path="config/experiments/exp_4_1_rrf_zs.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="Reciprocal Rank Fusion with zero-shot models"
    ),
    "exp_4_2_rrf_ft": ExperimentConfig(
        name="RRF (Fine-tuned)",
        model_type="rrf",
        variant="fine-tuned",
        stage=2,
        config_path="config/experiments/exp_4_2_rrf_ft.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="Reciprocal Rank Fusion with fine-tuned models"
    ),
    "exp_4_3_l2r_zs": ExperimentConfig(
        name="LightGBM L2R (Zero-shot)",
        model_type="l2r",
        variant="zero-shot",
        stage=3,
        config_path="config/experiments/exp_4_3_l2r_zs.yaml",
        requires_training=True,
        training_time_estimate="1-2h",
        description="LightGBM Learning-to-Rank with zero-shot features"
    ),
    "exp_4_4_l2r_ft": ExperimentConfig(
        name="LightGBM L2R (Fine-tuned)",
        model_type="l2r",
        variant="fine-tuned",
        stage=3,
        config_path="config/experiments/exp_4_4_l2r_ft.yaml",
        requires_training=True,
        training_time_estimate="1-2h",
        description="LightGBM Learning-to-Rank with fine-tuned features"
    ),
    
    # Track 5: Multi-Stage Pipeline
    "exp_5_1_pipeline_basic": ExperimentConfig(
        name="Multi-Stage Pipeline (Basic)",
        model_type="pipeline",
        variant="basic",
        stage=0,  # All stages
        config_path="config/experiments/exp_5_1_pipeline_basic.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="Basic multi-stage pipeline"
    ),
    "exp_5_2_pipeline_optimized": ExperimentConfig(
        name="Multi-Stage Pipeline (Optimized)",
        model_type="pipeline",
        variant="optimized",
        stage=0,
        config_path="config/experiments/exp_5_2_pipeline_optimized.yaml",
        requires_training=False,
        training_time_estimate="10-12h",
        description="Optimized multi-stage pipeline with fine-tuned models"
    ),
    
    # Track 6: Improvements
    "exp_6_1_query_enhancement": ExperimentConfig(
        name="Query Enhancement (Exp 6.1)",
        model_type="pipeline",
        variant="query_enhanced",
        stage=0,
        config_path="config/experiments/exp_6_1_query_enhancement.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="Pipeline with query enhancement (citation_context + source_paper_title + abstract)"
    ),
    "exp_6_1b_1_context_before": ExperimentConfig(
        name="Context Enhancement - Before (Exp 6.1b.1)",
        model_type="pipeline",
        variant="context_before",
        stage=0,
        config_path="config/experiments/exp_6_1b_1_context_before.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="Pipeline with context_before + citation_context"
    ),
    "exp_6_1b_2_context_after": ExperimentConfig(
        name="Context Enhancement - After (Exp 6.1b.2)",
        model_type="pipeline",
        variant="context_after",
        stage=0,
        config_path="config/experiments/exp_6_1b_2_context_after.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="Pipeline with citation_context + context_after"
    ),
    "exp_6_1b_3_context_both": ExperimentConfig(
        name="Context Enhancement - Both (Exp 6.1b.3)",
        model_type="pipeline",
        variant="context_both",
        stage=0,
        config_path="config/experiments/exp_6_1b_3_context_both.yaml",
        requires_training=False,
        training_time_estimate="0h",
        description="Pipeline with context_before + citation_context + context_after"
    ),
}

def get_experiment_config(exp_id: str) -> ExperimentConfig:
    """获取实验配置"""
    return ALL_EXPERIMENTS.get(exp_id)

def list_experiments_by_track(track: int) -> List[ExperimentConfig]:
    """按track列出实验"""
    track_map = {
        1: ["exp_1_1", "exp_1_2", "exp_1_3"],
        2: ["exp_2_1", "exp_2_2", "exp_2_3", "exp_2_4"],
        3: ["exp_3_1", "exp_3_2", "exp_3_3"],
        4: ["exp_4_1", "exp_4_2", "exp_4_3", "exp_4_4"],
        5: ["exp_5_1", "exp_5_2"]
    }
    
    prefixes = track_map.get(track, [])
    # 使用前缀匹配找到所有匹配的实验
    matching_exps = []
    for exp_id, exp_config in ALL_EXPERIMENTS.items():
        if any(exp_id.startswith(prefix) for prefix in prefixes):
            matching_exps.append(exp_config)
    
    return matching_exps

def list_experiments_by_variant(variant: str) -> List[ExperimentConfig]:
    """按variant列出实验"""
    return [exp for exp in ALL_EXPERIMENTS.values() if exp.variant == variant]

