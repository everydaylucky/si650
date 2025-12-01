# 数据集重新组织指南

## 📊 新数据格式

为了便于索引，我们将数据重新组织为两个独立文件：

### 1. `corpus.json` - 索引文档文件
包含所有唯一文档，用于构建检索索引。

**格式**：
```json
[
  {
    "id": "2009.05166",
    "paper_id": "2009.05166",
    "title": "FILTER: An Enhanced Fusion Method...",
    "abstract": "...",
    "categories": ["cs.CL"],
    "year": 2020
  },
  ...
]
```

**特点**：
- ✅ 包含所有唯一文档（4504个）
- ✅ 去重，避免重复索引
- ✅ 可以复用，多个实验共享同一索引

### 2. `test.json` - 测试数据文件
只包含查询和 ground truth，不包含所有 candidates。

**格式**：
```json
[
  {
    "sample_id": "2010.11934_cite_00000",
    "source_paper": {
      "id": "2010.11934",
      "title": "...",
      "abstract": "...",
      "categories": ["cs.CL"],
      "year": 2020
    },
    "citation_context": {
      "text": "Metrics for XLM...",
      "context_before": "",
      "context_after": "All other metrics...",
      "section": "Results"
    },
    "target_paper_id": "2009.05166"
  },
  ...
]
```

**特点**：
- ✅ 文件更小，加载更快
- ✅ 只包含必要信息（查询 + ground truth）
- ✅ 清晰的分离：索引 vs 评估

## 🚀 使用方法

### 1. 重新组织数据

```bash
# 从 data/full 格式转换
python scripts/reorganize_data_for_indexing.py \
    --input data/full/test.json \
    --output data/full_indexed
```

**输出**：
- `data/full_indexed/corpus.json` - 索引文档
- `data/full_indexed/test.json` - 测试数据
- `data/full_indexed/metadata.json` - 元数据

### 2. 运行实验

使用新格式运行实验：

```bash
# 使用新格式数据
python scripts/run_all_experiments.py \
    --experiment exp_6_1b_1_context_before \
    --data_dir data/full_indexed
```

**代码会自动检测**：
- 如果存在 `corpus.json` 和 `test.json`，使用新格式
- 否则，使用传统格式（从测试数据中提取文档）

## 📈 优势对比

| 特性 | 旧格式 | 新格式 |
|------|--------|--------|
| **索引文档** | 从测试数据中提取 | 独立的 `corpus.json` |
| **文件大小** | 278MB (test.json) | ~50MB (corpus) + ~10MB (test) |
| **加载速度** | 慢（需要解析所有 candidates） | 快（只加载必要数据） |
| **索引复用** | ❌ 每次重新构建 | ✅ 可以复用 |
| **数据分离** | ❌ 混合在一起 | ✅ 清晰的分离 |
| **去重** | ⚠️ 需要手动处理 | ✅ 自动去重 |

## 🔄 数据流程

### 旧格式流程
```
test.json (278MB)
  ↓
解析所有 candidates (188,800个)
  ↓
提取唯一文档 (4,504个)
  ↓
构建索引
  ↓
评估
```

### 新格式流程
```
corpus.json (50MB) → 构建索引 (一次性)
test.json (10MB)   → 评估 (快速加载)
```

## 📝 文件结构

```
data/
├── full/                    # 原始数据
│   ├── test.json           # 278MB (包含所有 candidates)
│   ├── val.json
│   └── train.json
│
└── full_indexed/            # 重新组织后的数据
    ├── corpus.json         # 50MB (所有唯一文档)
    ├── test.json           # 10MB (只包含查询和 ground truth)
    └── metadata.json       # 元数据
```

## ✅ 验证

运行验证脚本：

```bash
python -c "
import json
from pathlib import Path

corpus_file = Path('data/full_indexed/corpus.json')
test_file = Path('data/full_indexed/test.json')

# 检查文件
with open(corpus_file) as f:
    corpus = json.load(f)
print(f'✓ corpus.json: {len(corpus)} 个文档')

with open(test_file) as f:
    test_data = json.load(f)
print(f'✓ test.json: {len(test_data)} 个样本')

# 验证 ground truth 都在索引中
corpus_ids = {doc['id'] for doc in corpus}
test_ids = {s['target_paper_id'] for s in test_data if s.get('target_paper_id')}
missing = test_ids - corpus_ids

if missing:
    print(f'⚠️  警告: {len(missing)} 个 ground truth 不在索引中')
else:
    print(f'✓ 所有 ground truth 都在索引中')
"
```

## 🎯 下一步

1. **重新组织所有数据**：
   ```bash
   # 重新组织 train, val, test
   python scripts/reorganize_data_for_indexing.py --input data/full/train.json --output data/full_indexed_train
   python scripts/reorganize_data_for_indexing.py --input data/full/val.json --output data/full_indexed_val
   python scripts/reorganize_data_for_indexing.py --input data/full/test.json --output data/full_indexed
   ```

2. **运行实验**：
   ```bash
   python scripts/run_all_experiments.py --experiment exp_6_1b_1_context_before --data_dir data/full_indexed
   ```

3. **对比性能**：
   - 使用新格式应该更快
   - 索引可以复用，节省时间
   - 评估结果应该一致

