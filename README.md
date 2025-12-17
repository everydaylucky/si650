# Final Test - Citation Recommendation System

Implementation of a citation recommendation system based on multi-stage architecture.

## Project Structure

```
final_test/
├── config/              # Configuration files
├── data/                # Data directory
├── src/                 # Source code
│   ├── models/         # Model implementations
│   ├── pipeline/       # Multi-stage pipeline
│   ├── features/       # Feature extraction
│   ├── evaluation/     # Evaluation framework
│   └── utils/          # Utility functions
├── scripts/            # Execution scripts
├── tests/              # Test files
└── experiments/        # Experimental results
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Data Location

Data should be placed in the `data/processed/` directory:
- `train.json` - Training set
- `val.json` - Validation set  
- `test.json` - Test set

### Quick Start

1. **Use example data** (for testing):
   ```bash
   # Example data is included in data/processed/example_*.json
   ```

2. **Convert from existing data**:
   ```bash
   python scripts/prepare_data.py convert <input_file> data/processed/train.json
   ```

3. **Check data quality**:
   ```bash
   python scripts/prepare_data.py check data/processed/train.json
   ```

For detailed data format documentation, see:
- `DATA_QUICK_START.md` - Quick reference
- `data/DATA_FORMAT.md` - Complete format documentation

## Quick Start

### 1. Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_retrievers.py
```

### 2. Use Retrievers

```python
from src.models.retrieval import BM25Retriever

# Create retriever
retriever = BM25Retriever()

# Build index
documents = [
    {"id": "1", "title": "Paper 1", "abstract": "Abstract 1"},
    {"id": "2", "title": "Paper 2", "abstract": "Abstract 2"}
]
retriever.build_index(documents)

# Retrieve
results = retriever.retrieve("query text", top_k=10)
```

### 3. Use Multi-Stage Pipeline

```python
import yaml
from src.pipeline import MultiStagePipeline

# Load configuration
with open("config/model_config.yaml") as f:
    config = yaml.safe_load(f)

# Create pipeline
pipeline = MultiStagePipeline(config)

# Build indices
pipeline.build_indices(documents)

# Retrieve
query = {"citation_context": "Recent work shows..."}
results = pipeline.retrieve(query)
```

## Model Description

### Stage 1: Initial Retrieval
- **BM25Retriever**: BM25 sparse retrieval
- **TFIDFRetriever**: TF-IDF retrieval
- **DenseRetriever**: SPECTER2 dense retrieval (requires GPU)

### Stage 2: Re-ranking
- **ReciprocalRankFusion**: RRF fusion
- **BiEncoder**: SciBERT bi-encoder (requires GPU)

### Stage 3: Final Ranking
- **CrossEncoderRanker**: Cross-Encoder ranking (requires GPU)
- **L2RRanker**: LightGBM Learning-to-Rank

## Evaluation Metrics

- MRR (Mean Reciprocal Rank)
- Recall@K (K=5, 10, 20, 50)
- Precision@K (K=10, 20)
- NDCG@K (K=10, 20)

## Testing

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_retrievers
```

## Configuration

Edit `config/model_config.yaml` to configure the models and hyperparameters to use.

## Running Experiments

### Experiment Overview

The project includes 20+ experiments organized into 6 tracks:

- **Track 1**: Traditional IR Baselines (BM25, TF-IDF, PRF)
- **Track 2**: Zero-shot Dense Models (SciBERT, SPECTER2, ColBERT, Cross-Encoder)
- **Track 3**: Fine-tuned Models (SciBERT, SPECTER2, Cross-Encoder)
- **Track 4**: Fusion Methods (RRF, LightGBM L2R)
- **Track 5**: Multi-Stage Pipeline (Basic, Optimized)
- **Track 6**: Query/Context Enhancements

### Quick Start: Run a Single Experiment

```bash
# Run BM25 baseline
python scripts/run_all_experiments.py \
    --experiment exp_1_1_bm25 \
    --data_dir data/processed/fast_experiment

# Run SPECTER2 zero-shot
python scripts/run_all_experiments.py \
    --experiment exp_2_2_specter2_zs \
    --data_dir data/processed/fast_experiment

# Run optimized pipeline
python scripts/run_all_experiments.py \
    --experiment exp_5_2_pipeline_optimized \
    --data_dir data/processed/fast_experiment
```

### Run Experiments by Track

```bash
# Run all Track 1 experiments (baselines)
python scripts/run_all_experiments.py \
    --track 1 \
    --data_dir data/processed/fast_experiment

# Run all Track 2 experiments (zero-shot models)
python scripts/run_all_experiments.py \
    --track 2 \
    --data_dir data/processed/fast_experiment
```

### Run All Experiments

```bash
# Run all experiments (requires trained models)
python scripts/run_all_experiments.py \
    --all \
    --data_dir data/processed/fast_experiment
```

### Fast Mode (Quick Testing)

For quick testing with a smaller sample:

```bash
# Fast mode: sample 472 instances
python scripts/run_all_experiments.py \
    --experiment exp_1_1_bm25 \
    --data_dir data/processed/fast_experiment \
    --fast

# Custom sample size
python scripts/run_all_experiments.py \
    --experiment exp_1_1_bm25 \
    --data_dir data/processed/fast_experiment \
    --sample_size 100

# Custom sample ratio
python scripts/run_all_experiments.py \
    --experiment exp_1_1_bm25 \
    --data_dir data/processed/fast_experiment \
    --sample_ratio 0.1
```

### Using Full Dataset

For experiments with the full dataset:

```bash
# Use full indexed dataset (separated corpus and test files)
python scripts/run_all_experiments.py \
    --experiment exp_5_2_pipeline_optimized \
    --data_dir data/full_indexed
```

### Training Models

Some experiments require training models first. The training will be done automatically when running experiments, but you can also train separately:

```bash
# Train SciBERT bi-encoder
python scripts/train_scibert.py \
    --train_file data/processed/fast_experiment/train.json \
    --val_file data/processed/fast_experiment/val.json \
    --output_dir experiments/checkpoints/scibert \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5

# Train SPECTER2
python scripts/train_specter2.py \
    --train_file data/processed/fast_experiment/train.json \
    --val_file data/processed/fast_experiment/val.json \
    --output_dir experiments/checkpoints/specter2 \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5

# Train Cross-Encoder
python scripts/train_cross_encoder.py \
    --train_file data/processed/fast_experiment/train.json \
    --val_file data/processed/fast_experiment/val.json \
    --output_dir experiments/checkpoints/cross_encoder \
    --epochs 3 \
    --batch_size 32 \
    --learning_rate 2e-5

# Train LightGBM L2R
python scripts/train_l2r.py \
    --train_file data/processed/fast_experiment/train.json \
    --val_file data/processed/fast_experiment/val.json \
    --output_dir experiments/checkpoints/l2r/zs \
    --variant zero-shot
```

### Viewing Results

#### Analyze All Results

```bash
# View all experiment results
python scripts/analyze_results.py
```

#### Compare Specific Experiments

```bash
# Compare two experiments
python scripts/analyze_results.py \
    --compare exp_2_1_scibert_zs exp_3_1_scibert_ft

# Compare multiple experiments
python scripts/analyze_results.py \
    --compare exp_1_1_bm25 exp_2_2_specter2_zs exp_5_2_pipeline_optimized
```

#### Filter by Model Type or Variant

```bash
# View all SciBERT experiments
python scripts/analyze_results.py --model_type scibert

# View all zero-shot experiments
python scripts/analyze_results.py --variant zero-shot

# View all fine-tuned experiments
python scripts/analyze_results.py --variant fine-tuned
```

### Results Location

All experiment results are saved in:
- `experiments/results/` - JSON files with detailed results
- `experiments/results/all_experiments.json` - Master file with all experiments
- `experiments/results/experiment_summary.csv` - Summary table
- `experiments/results/analysis_report.md` - Analysis report

Each experiment result includes:
- Metrics: MRR, Recall@K, NDCG@K, Precision@K
- Configuration used
- Training information (if applicable)
- Timestamp and experiment metadata

### Experiment List

All available experiments are defined in `src/experiments/experiment_config.py`. Key experiments include:

- `exp_1_1_bm25` - BM25 Baseline
- `exp_1_2_tfidf` - TF-IDF Baseline
- `exp_2_2_specter2_zs` - SPECTER2 Zero-shot
- `exp_3_1_scibert_ft` - SciBERT Fine-tuned
- `exp_4_3_l2r_zs` - LightGBM L2R Zero-shot
- `exp_4_4_l2r_ft` - LightGBM L2R Fine-tuned
- `exp_5_2_pipeline_optimized` - Optimized Multi-Stage Pipeline
- `exp_6_1_query_enhancement` - Query Enhancement
- `exp_6_1b_1_context_before` - Context Enhancement (Before)
- `exp_6_1b_2_context_after` - Context Enhancement (After)
- `exp_6_1b_3_context_both` - Context Enhancement (Both)

See `src/experiments/experiment_config.py` for the complete list.

### Running Context Enhancement Experiments

```bash
# Run all context enhancement experiments
bash scripts/run_all_fast_experiments.sh
```

Or individually:

```bash
python scripts/run_all_experiments.py --experiment exp_6_1_query_enhancement --data_dir data/processed/fast_experiment
python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/processed/fast_experiment
python scripts/run_all_experiments.py --experiment exp_6_1b_2_context_after --data_dir data/processed/fast_experiment
python scripts/run_all_experiments.py --experiment exp_6_1b_3_context_both --data_dir data/processed/fast_experiment
```

## Notes

1. **GPU Models**: SPECTER2, BiEncoder, and CrossEncoder require GPU. If you don't have a GPU, you can disable them in the configuration file.
2. **Data Format**: Documents need to contain `id`, `title`, and `abstract` fields.
3. **Index Building**: Index building is required on first use and may take some time.
4. **Training Time**: Fine-tuning experiments require significant time (3-6 hours per model).
5. **Data Location**: Use `data/processed/fast_experiment` for quick testing, `data/full_indexed` for full evaluation.

## Development Plan

- [x] Basic retriever implementation
- [x] Multi-stage pipeline
- [x] Evaluation framework
- [x] Test files
- [x] Training scripts
- [x] Fine-tuning support
- [x] Complete feature extractor integration
- [x] Experiment management system

## License

MIT License
