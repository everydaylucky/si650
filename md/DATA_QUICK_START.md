# 数据快速开始指南

## 📁 数据应该放在哪里？

```
final_test/
└── data/
    └── processed/
        ├── train.json    ← 训练集（必需）
        ├── val.json      ← 验证集（必需）
        └── test.json     ← 测试集（必需）
```

## 📋 数据格式示例

### 最小示例（一个样本）

```json
{
  "citation_context": "Recent work shows that transformer models...",
  "source_paper_id": "1910.10683",
  "target_paper_id": "1706.03762",
  "source_paper": {
    "id": "1910.10683",
    "title": "Exploring the Limits of Transfer Learning...",
    "abstract": "Transfer learning, where a model...",
    "categories": ["cs.LG", "cs.CL"],
    "year": 2019
  },
  "target_paper": {
    "id": "1706.03762",
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models...",
    "categories": ["cs.CL", "cs.LG"],
    "year": 2017
  }
}
```

### 完整示例（训练集需要负样本）

```json
{
  "citation_context": "...",
  "source_paper_id": "...",
  "target_paper_id": "...",
  "source_paper": {...},
  "target_paper": {...},
  "negatives": [        // 仅训练集需要
    {
      "id": "...",
      "title": "...",
      "abstract": "...",
      "categories": [...],
      "year": 2015
    }
  ],
  "metadata": {         // 可选
    "section": "Introduction",
    "source_year": 2019,
    "target_year": 2017
  }
}
```

## 🚀 快速开始

### 1. 使用示例数据（测试用）

```bash
# 示例数据已准备好
ls data/processed/example_*.json
```

### 2. 从SI650项目转换数据

```bash
# 如果你有SI650的citation数据
python scripts/prepare_data.py convert \
    ../other/citation_ground_truth.json \
    data/processed/train.json
```

### 3. 检查数据

```bash
python scripts/prepare_data.py check data/processed/train.json
```

## 📊 数据要求

| 数据集 | 样本数 | 负样本比例 | 文件位置 |
|--------|--------|------------|----------|
| 训练集 | 12,844 | 1:10 | `data/processed/train.json` |
| 验证集 | 1,605 | 1:99 | `data/processed/val.json` |
| 测试集 | 1,606 | 1:99 | `data/processed/test.json` |

## ✅ 必需字段检查清单

每个样本必须包含：

- [x] `citation_context` - 引用上下文文本
- [x] `source_paper_id` - 源论文ID
- [x] `target_paper_id` - 目标论文ID
- [x] `source_paper.id` - 源论文ID
- [x] `source_paper.title` - 源论文标题
- [x] `source_paper.abstract` - 源论文摘要
- [x] `target_paper.id` - 目标论文ID
- [x] `target_paper.title` - 目标论文标题
- [x] `target_paper.abstract` - 目标论文摘要

## 📝 详细文档

- **完整格式说明**: `data/DATA_FORMAT.md`
- **数据目录说明**: `data/README.md`
- **示例数据**: `data/processed/example_*.json`

## ⚠️ 注意事项

1. **时间一致性**: 源论文年份必须 ≥ 目标论文年份
2. **文本质量**: citation_context长度 ≥ 10个单词
3. **文件编码**: 必须使用UTF-8编码
4. **JSON格式**: 必须是有效的JSON数组

## 🔍 验证数据

运行测试确保数据格式正确：

```bash
python -c "
from src.utils import load_json
data = load_json('data/processed/example_train.json')
print(f'✓ 成功加载 {len(data)} 个样本')
print(f'✓ 第一个样本的citation_context: {data[0][\"citation_context\"][:50]}...')
"
```

