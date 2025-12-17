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

## Notes

1. **GPU Models**: SPECTER2, BiEncoder, and CrossEncoder require GPU. If you don't have a GPU, you can disable them in the configuration file.
2. **Data Format**: Documents need to contain `id`, `title`, and `abstract` fields.
3. **Index Building**: Index building is required on first use and may take some time.

## Development Plan

- [x] Basic retriever implementation
- [x] Multi-stage pipeline
- [x] Evaluation framework
- [x] Test files
- [ ] Training scripts
- [ ] Fine-tuning support
- [ ] Complete feature extractor integration

## License

MIT License

# si650
