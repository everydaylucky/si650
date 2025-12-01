# 数据目录说明

## 目录结构

```
data/
├── raw/                    # 原始数据（可选）
│   └── citation_ground_truth.json  # 从SI650/other提取的原始数据
│
├── processed/              # 处理后的数据（必需）
│   ├── train.json         # 训练集（12,844个样本）
│   ├── val.json           # 验证集（1,605个样本）
│   └── test.json          # 测试集（1,606个样本）
│   └── example_train.json # 示例训练数据（用于测试）
│   └── example_test.json  # 示例测试数据（用于测试）
│
├── indices/               # 预构建的索引（自动生成，无需手动创建）
│   ├── bm25_index.pkl
│   ├── tfidf_index.pkl
│   └── faiss_index.bin
│
└── cache/                 # 缓存文件（自动生成）
```

## 快速开始

### 1. 准备数据

#### 方法A：使用示例数据（快速测试）

示例数据已包含在 `processed/example_train.json` 和 `example_test.json`，可以直接用于测试。

#### 方法B：从现有数据转换

如果你有SI650项目的citation数据：

```bash
# 转换数据
python scripts/prepare_data.py convert \
    ../other/citation_ground_truth.json \
    data/processed/train.json

# 检查数据质量
python scripts/prepare_data.py check data/processed/train.json
```

#### 方法C：手动准备

参考 `DATA_FORMAT.md` 中的格式说明，手动创建JSON文件。

### 2. 数据格式

每个JSON文件是一个数组，包含多个样本。每个样本格式：

```json
{
  "citation_context": "引用上下文文本",
  "source_paper_id": "源论文ID",
  "target_paper_id": "目标论文ID",
  "source_paper": {
    "id": "...",
    "title": "...",
    "abstract": "...",
    "categories": ["cs.LG"],
    "year": 2019
  },
  "target_paper": {
    "id": "...",
    "title": "...",
    "abstract": "...",
    "categories": ["cs.CL"],
    "year": 2017
  },
  "negatives": [...],  // 仅训练集需要
  "metadata": {...}    // 可选
}
```

详细格式说明见 `DATA_FORMAT.md`

### 3. 数据要求

- **训练集**: 建议12,844个样本，包含10个负样本/正样本
- **验证集**: 建议1,605个样本，包含99个负样本/正样本
- **测试集**: 建议1,606个样本，包含99个负样本/正样本

### 4. 数据检查

运行数据质量检查：

```bash
python scripts/prepare_data.py check data/processed/train.json
```

检查项包括：
- 必需字段完整性
- 时间一致性（源论文年份 ≥ 目标论文年份）
- 文本质量

## 注意事项

1. **文件编码**: 所有JSON文件必须使用UTF-8编码
2. **文件大小**: 如果数据很大，考虑使用JSONL格式（每行一个JSON）
3. **索引缓存**: 构建的索引会自动保存在 `indices/` 目录
4. **数据路径**: 代码中默认使用 `data/processed/` 下的文件

## 数据来源

如果你有SI650项目的citation数据，通常位于：
- `SI650/other/` - 数据提取脚本
- `SI650/SI-650-proj/data/` - 已处理的数据

可以使用 `scripts/prepare_data.py` 进行格式转换。

