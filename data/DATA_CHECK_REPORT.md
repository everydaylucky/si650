# 数据检查报告

## 📊 数据概览

### 原始数据格式

你的原始数据文件格式为：
```json
{
  "split": "test",
  "metadata": {...},
  "samples": [...]
}
```

每个样本格式：
```json
{
  "sample_id": "...",
  "source_paper": {...},
  "citation_context": {
    "text": "...",
    "context_before": "...",
    "context_after": "...",
    "section": "..."
  },
  "candidates": [
    {
      "arxiv_id": "...",
      "title": "...",
      "abstract": "...",
      "categories": "...",
      "label": 1,  // 1=positive, 0=negative
      "type": "positive"
    }
  ]
}
```

### 数据统计

| 数据集 | 原始样本数 | 转换后样本数 | 负样本比例 |
|--------|-----------|-------------|-----------|
| 训练集 | 12,587 | 12,587 | 1:10 |
| 验证集 | 1,580 | 1,580 | 1:99 |
| 测试集 | 1,888 | 1,888 | 1:99 |

## ✅ 转换结果

### 转换后的格式

转换后的数据符合final_test要求的格式：

```json
{
  "citation_context": "文本内容",
  "source_paper_id": "...",
  "target_paper_id": "...",
  "source_paper": {
    "id": "...",
    "title": "...",
    "abstract": "...",
    "categories": [...],
    "year": 2020
  },
  "target_paper": {
    "id": "...",
    "title": "...",
    "abstract": "...",
    "categories": [...],
    "year": 2020
  },
  "negatives": [...],  // 仅训练集
  "metadata": {...}
}
```

### 转换验证

✅ **所有必需字段存在**:
- `citation_context` (字符串)
- `source_paper_id`
- `target_paper_id`
- `source_paper` (包含 id, title, abstract)
- `target_paper` (包含 id, title, abstract)

✅ **字段类型正确**

✅ **训练集包含negatives字段** (10个负样本)

✅ **所有样本成功转换** (无跳过)

## 📝 下一步操作

### 1. 替换原文件（推荐）

转换后的文件已保存为 `*.converted`，如果验证无误，可以替换原文件：

```bash
cd data/processed/

# 备份原文件（已自动备份到 backup_original/）
# 替换为转换后的文件
mv train.json.converted train.json
mv val.json.converted val.json
mv test.json.converted test.json
```

### 2. 验证转换后的数据

```bash
# 使用检查脚本验证
python scripts/check_data_format.py

# 或者手动检查
python -c "
from src.utils import load_json
train = load_json('data/processed/train.json')
print(f'训练集: {len(train)} 个样本')
print(f'第一个样本字段: {list(train[0].keys())}')
"
```

### 3. 使用数据

转换后的数据可以直接用于final_test项目：

```python
from src.utils import load_json
from src.pipeline import MultiStagePipeline
import yaml

# 加载数据
train_data = load_json("data/processed/train.json")
test_data = load_json("data/processed/test.json")

# 准备所有文档（用于构建索引）
all_documents = []
seen_ids = set()
for sample in train_data + test_data:
    for paper in [sample["source_paper"], sample["target_paper"]]:
        if paper["id"] not in seen_ids:
            all_documents.append(paper)
            seen_ids.add(paper["id"])

# 创建管道并构建索引
with open("config/model_config.yaml") as f:
    config = yaml.safe_load(f)

pipeline = MultiStagePipeline(config)
pipeline.build_indices(all_documents)

# 使用查询
query = {
    "citation_context": train_data[0]["citation_context"],
    "source_paper_id": train_data[0]["source_paper_id"],
    "source_categories": train_data[0]["source_paper"]["categories"],
    "source_year": train_data[0]["source_paper"]["year"]
}

results = pipeline.retrieve(query)
```

## ⚠️ 注意事项

1. **原文件备份**: 原文件已自动备份到 `data/processed/backup_original/`
2. **文件大小**: 转换后的文件可能比原文件稍大（因为格式更详细）
3. **内存使用**: 加载大型JSON文件时注意内存使用
4. **索引构建**: 首次使用需要构建索引，可能需要一些时间

## 📈 数据质量

根据检查脚本的结果：

- ✅ 所有样本都有完整的citation_context
- ✅ 所有样本都有source_paper和target_paper
- ✅ 时间一致性（源论文年份 ≥ 目标论文年份）
- ✅ 文本质量良好（citation_context长度 ≥ 10单词）

## 🔄 如果需要重新转换

如果转换后的数据有问题，可以：

1. 恢复原文件：
   ```bash
   cp data/processed/backup_original/train.json data/processed/train.json
   ```

2. 重新运行转换：
   ```bash
   python scripts/convert_data_format.py
   ```

---

**转换完成时间**: 2024-11-24  
**转换脚本**: `scripts/convert_data_format.py`  
**检查脚本**: `scripts/check_data_format.py`


