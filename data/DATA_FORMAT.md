# 数据格式说明

本文档说明final_test项目需要的数据格式和存放位置。

## 数据目录结构

```
data/
├── raw/                    # 原始数据（可选）
├── processed/              # 处理后的数据（必需）
│   ├── train.json         # 训练集
│   ├── val.json           # 验证集
│   └── test.json          # 测试集
├── indices/               # 预构建的索引（自动生成）
│   ├── bm25_index.pkl
│   ├── tfidf_index.pkl
│   └── faiss_index.bin
└── cache/                 # 缓存文件（自动生成）
```

## 数据格式

### 训练/验证/测试集格式

每个数据集文件（`train.json`, `val.json`, `test.json`）都是一个JSON数组，每个元素代表一个查询样本。

#### 基本格式

```json
[
  {
    "citation_context": "Recent work shows that transformer models...",
    "source_paper_id": "1910.10683",
    "source_paper": {
      "id": "1910.10683",
      "title": "Exploring the Limits of Transfer Learning...",
      "abstract": "Transfer learning, where a model is first trained...",
      "categories": ["cs.LG", "cs.CL", "stat.ML"],
      "year": 2019
    },
    "target_paper_id": "1706.03762",
    "target_paper": {
      "id": "1706.03762",
      "title": "Attention Is All You Need",
      "abstract": "The dominant sequence transduction models...",
      "categories": ["cs.CL", "cs.LG"],
      "year": 2017
    },
    "negatives": [
      {
        "id": "1508.05326",
        "title": "Some Other Paper",
        "abstract": "This is a negative example...",
        "categories": ["cs.CV"],
        "year": 2015
      }
    ],
    "metadata": {
      "section": "Introduction",
      "source_year": 2019,
      "target_year": 2017
    }
  }
]
```

### 字段说明

#### 必需字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `citation_context` | string | 引用上下文文本（包含[CITE]标记或已替换） |
| `source_paper_id` | string | 源论文ID（ground truth） |
| `target_paper_id` | string | 目标论文ID（ground truth） |
| `source_paper` | object | 源论文完整信息 |
| `target_paper` | object | 目标论文完整信息 |

#### 源论文/目标论文对象格式

```json
{
  "id": "1910.10683",                    // 必需：论文唯一ID
  "title": "Paper Title",                 // 必需：论文标题
  "abstract": "Paper abstract text...",   // 必需：论文摘要
  "categories": ["cs.LG", "cs.CL"],       // 可选：类别列表
  "year": 2019                            // 可选：发表年份
}
```

#### 负样本（仅训练集需要）

训练集需要包含负样本，格式与`target_paper`相同：

```json
"negatives": [
  {
    "id": "1508.05326",
    "title": "Negative Example Paper",
    "abstract": "...",
    "categories": ["cs.CV"],
    "year": 2015
  }
]
```

**注意**：
- 训练集：通常包含10个负样本（1:10比例）
- 验证/测试集：通常包含99个负样本（1:99比例）

#### 元数据（可选）

```json
"metadata": {
  "section": "Introduction",      // citation所在section
  "source_year": 2019,            // 源论文年份
  "target_year": 2017,            // 目标论文年份
  "source_categories": ["cs.LG"]   // 源论文类别（用于特征提取）
}
```

## 数据准备

### 方法1：从现有数据转换

如果你已经有citation数据，可以使用以下脚本转换：

```python
# scripts/prepare_data.py (需要创建)
import json

def convert_to_final_test_format(input_file, output_file):
    # 读取原始数据
    with open(input_file) as f:
        data = json.load(f)
    
    # 转换格式
    converted = []
    for sample in data['samples']:
        # 只取第一个target paper（简化示例）
        target = sample['target_papers'][0]
        
        converted_sample = {
            "citation_context": sample['citation_context']['text'],
            "source_paper_id": sample['source_paper']['arxiv_id'],
            "target_paper_id": target['arxiv_id'],
            "source_paper": {
                "id": sample['source_paper']['arxiv_id'],
                "title": sample['source_paper']['title'],
                "abstract": sample['source_paper']['abstract'],
                "categories": sample['source_paper']['categories'].split(),
                "year": int(sample['source_paper'].get('year', 2020))
            },
            "target_paper": {
                "id": target['arxiv_id'],
                "title": target['title'],
                "abstract": target['abstract'],
                "categories": target['categories'].split(),
                "year": int(target.get('year', 2020))
            },
            "metadata": {
                "section": sample['citation_context'].get('section', 'Unknown'),
                "source_year": int(sample['source_paper'].get('year', 2020)),
                "target_year": int(target.get('year', 2020))
            }
        }
        converted.append(converted_sample)
    
    # 保存
    with open(output_file, 'w') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
```

### 方法2：手动创建示例数据

见 `data/processed/example_train.json`

## 数据统计要求

根据设计文档：

- **训练集**: 12,844个样本，297篇源论文，1:10负样本比例
- **验证集**: 1,605个样本，38篇源论文，1:99负样本比例
- **测试集**: 1,606个样本，37篇源论文，1:99负样本比例

## 数据质量要求

1. **时间一致性**: 源论文年份 ≥ 目标论文年份（无时间泄漏）
2. **文本质量**: citation_context长度 ≥ 10个单词
3. **完整性**: 所有必需字段必须存在
4. **唯一性**: 每个样本的source_paper_id和target_paper_id组合唯一

## 使用数据

### 加载数据

```python
from src.utils import load_json

# 加载训练集
train_data = load_json("data/processed/train.json")
print(f"训练样本数: {len(train_data)}")

# 加载测试集
test_data = load_json("data/processed/test.json")
```

### 在管道中使用

```python
from src.pipeline import MultiStagePipeline
import yaml

# 加载配置
with open("config/model_config.yaml") as f:
    config = yaml.safe_load(f)

# 创建管道
pipeline = MultiStagePipeline(config)

# 准备所有文档（用于构建索引）
all_documents = []
for sample in train_data + test_data:
    all_documents.append(sample["target_paper"])
    if sample["target_paper"]["id"] not in [d["id"] for d in all_documents]:
        all_documents.append(sample["target_paper"])

# 构建索引
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

## 注意事项

1. **文件编码**: 所有JSON文件使用UTF-8编码
2. **文件大小**: 大型数据集建议使用JSONL格式（每行一个JSON对象）
3. **内存管理**: 如果数据太大，考虑分批加载
4. **索引缓存**: 构建好的索引会保存在`data/indices/`，可以重复使用

## 数据检查

运行数据检查脚本：

```bash
python scripts/check_data.py data/processed/train.json
```

检查项：
- 必需字段完整性
- 时间一致性
- 文本质量
- 数据统计信息

