# Final Test - Citation Recommendation System

基于多阶段架构的引用推荐系统实现。

## 项目结构

```
final_test/
├── config/              # 配置文件
├── data/                # 数据目录
├── src/                 # 源代码
│   ├── models/         # 模型实现
│   ├── pipeline/       # 多阶段管道
│   ├── features/       # 特征提取
│   ├── evaluation/     # 评估框架
│   └── utils/          # 工具函数
├── scripts/            # 执行脚本
├── tests/              # 测试文件
└── experiments/        # 实验结果
```

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 数据准备

### 数据位置

数据应放在 `data/processed/` 目录下：
- `train.json` - 训练集
- `val.json` - 验证集  
- `test.json` - 测试集

### 快速开始

1. **使用示例数据**（用于测试）：
   ```bash
   # 示例数据已包含在 data/processed/example_*.json
   ```

2. **从现有数据转换**：
   ```bash
   python scripts/prepare_data.py convert <input_file> data/processed/train.json
   ```

3. **检查数据质量**：
   ```bash
   python scripts/prepare_data.py check data/processed/train.json
   ```

详细数据格式说明见：
- `DATA_QUICK_START.md` - 快速参考
- `data/DATA_FORMAT.md` - 完整格式说明

## 快速开始

### 1. 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_retrievers.py
```

### 2. 使用检索器

```python
from src.models.retrieval import BM25Retriever

# 创建检索器
retriever = BM25Retriever()

# 构建索引
documents = [
    {"id": "1", "title": "Paper 1", "abstract": "Abstract 1"},
    {"id": "2", "title": "Paper 2", "abstract": "Abstract 2"}
]
retriever.build_index(documents)

# 检索
results = retriever.retrieve("query text", top_k=10)
```

### 3. 使用多阶段管道

```python
import yaml
from src.pipeline import MultiStagePipeline

# 加载配置
with open("config/model_config.yaml") as f:
    config = yaml.safe_load(f)

# 创建管道
pipeline = MultiStagePipeline(config)

# 构建索引
pipeline.build_indices(documents)

# 检索
query = {"citation_context": "Recent work shows..."}
results = pipeline.retrieve(query)
```

## 模型说明

### Stage 1: 初始检索
- **BM25Retriever**: BM25稀疏检索
- **TFIDFRetriever**: TF-IDF检索
- **DenseRetriever**: SPECTER2密集检索（需要GPU）

### Stage 2: 重排序
- **ReciprocalRankFusion**: RRF融合
- **BiEncoder**: SciBERT双编码器（需要GPU）

### Stage 3: 最终排序
- **CrossEncoderRanker**: Cross-Encoder排序（需要GPU）
- **L2RRanker**: LightGBM Learning-to-Rank

## 评估指标

- MRR (Mean Reciprocal Rank)
- Recall@K (K=5, 10, 20, 50)
- Precision@K (K=10, 20)
- NDCG@K (K=10, 20)

## 测试

```bash
# 运行所有测试
python -m unittest discover tests

# 运行特定测试文件
python -m unittest tests.test_retrievers
```

## 配置

编辑 `config/model_config.yaml` 来配置使用的模型和超参数。

## 注意事项

1. **GPU模型**: SPECTER2, BiEncoder, CrossEncoder需要GPU。如果没有GPU，可以在配置文件中关闭它们。
2. **数据格式**: 文档需要包含 `id`, `title`, `abstract` 字段。
3. **索引构建**: 首次使用需要构建索引，可能需要一些时间。

## 开发计划

- [x] 基础检索器实现
- [x] 多阶段管道
- [x] 评估框架
- [x] 测试文件
- [ ] 训练脚本
- [ ] Fine-tuning支持
- [ ] 完整特征提取器集成

## 许可证

MIT License

# si650
