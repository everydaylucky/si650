# Exp 6.1b 实验说明

## 📋 source_paper 内容

根据实际数据检查，`source_paper` 包含以下字段：

```json
{
  "id": "2010.11934",                    // 论文ID（如 arXiv ID）
  "title": "mT5: A massively multilingual pre-trained text-to-text transformer",  // 论文标题
  "abstract": "The recent Text-to-Text Transfer Transformer (T5)...",  // 论文摘要（可能很长）
  "categories": ["cs.CL"],              // 论文类别
  "year": 2020                          // 发表年份
}
```

### 关键字段说明

- **title**: 论文标题，通常包含论文的核心主题
- **abstract**: 论文摘要，包含论文的主要内容、方法、贡献等
- **categories**: 论文类别（如 cs.CL, cs.LG），表示研究领域

---

## 🔬 实验设计说明

### Exp 6.1b.1: 仅添加前后文

**查询组成**:
```
query = context_before + citation_context + context_after
```

**示例**:
```
context_before: "Recent work in NLP has shown that"
citation_context: "transformer models achieve state-of-the-art results"
context_after: "on various downstream tasks."

完整查询: "Recent work in NLP has shown that transformer models achieve state-of-the-art results on various downstream tasks."
```

**目的**: 验证前后文是否有助于理解引用句的语义

---

### Exp 6.1b.2: 前后文 + Source Paper

**查询组成**:
```
query = context_before + citation_context + context_after 
      + source_paper.title + source_paper.abstract[:200]
```

**示例**:
```
context_before: "Recent work in NLP has shown that"
citation_context: "transformer models achieve state-of-the-art results"
context_after: "on various downstream tasks."
source_paper.title: "mT5: A massively multilingual pre-trained text-to-text transformer"
source_paper.abstract[:200]: "The recent Text-to-Text Transfer Transformer (T5) leveraged a unified text-to-text format and scale to attain state-of-the-art results on a wide variety of English-language NLP tasks..."

完整查询: "Recent work in NLP has shown that transformer models achieve state-of-the-art results on various downstream tasks. mT5: A massively multilingual pre-trained text-to-text transformer The recent Text-to-Text Transfer Transformer (T5) leveraged a unified text-to-text format and scale to attain state-of-the-art results on a wide variety of English-language NLP tasks..."
```

**目的**: 
1. 前后文提供局部上下文
2. source_paper 提供全局上下文（源论文的主题和内容）
3. 两者结合，提供更完整的语义信息

---

## 💡 为什么添加 source_paper？

### 1. 提供主题上下文

**例子**:
- 如果 source_paper 是关于 "transformer" 的
- 那么 citation_context 很可能也在讨论 transformer 相关的内容
- 添加 source_paper 信息可以帮助模型理解引用句的主题

### 2. 减少歧义

**例子**:
- citation_context: "This method achieves good results"
- 如果不知道 source_paper，可能不清楚 "This method" 指什么
- 如果知道 source_paper 是关于 "BERT" 的，就能理解 "This method" 可能指 BERT

### 3. 增强语义匹配

**例子**:
- citation_context 可能只提到 "attention mechanism"
- source_paper 的 abstract 可能详细描述了 "self-attention"、"multi-head attention" 等
- 添加 source_paper 可以让检索模型更好地匹配相关论文

---

## 📊 实验对比

| 实验 | 查询组成 | 信息量 | 预期效果 |
|------|---------|--------|---------|
| 基线 | citation_context | 最小 | MRR = 0.3428 |
| Exp 6.1 | citation_context + source_paper | 中等 | MRR = 0.3414 (略降) |
| Exp 6.1b.1 | context_before + citation + context_after | 中等 | MRR = 0.35-0.37 (预期) |
| Exp 6.1b.2 | 6.1b.1 + source_paper | **最大** | MRR = 0.36-0.39 (预期) |

---

## ⚠️ 潜在问题

### 1. 查询过长

**问题**: 添加 source_paper.abstract 后，查询可能变得很长

**解决**: 
- 限制 abstract 长度（如 200 字符）
- 只使用 abstract 的前几句

### 2. 信息噪声

**问题**: source_paper 的信息可能包含不相关内容

**解决**:
- 只使用 title（通常最相关）
- 使用 abstract 的关键句子（需要提取）

### 3. 权重问题

**问题**: citation_context 应该是最重要的，但添加太多其他信息可能稀释其重要性

**解决**:
- 加权组合（citation_context 权重更高）
- 或者只在特定阶段使用 source_paper（如 Stage2/Stage3）

---

## 🎯 实施建议

### 方案1: 简单组合（当前 Exp 6.1 的方式）
```python
query = f"{context_before} {citation_context} {context_after} {source_title} {source_abstract[:200]}"
```

### 方案2: 加权组合（更精细）
```python
# citation_context 权重最高
query = f"{citation_context} {context_before} {context_after} {source_title} {source_abstract[:100]}"
```

### 方案3: 分阶段使用（更灵活）
```python
# Stage1: 只用 citation_context + 前后文
stage1_query = f"{context_before} {citation_context} {context_after}"

# Stage2/Stage3: 添加 source_paper
stage2_query = f"{stage1_query} {source_title} {source_abstract[:200]}"
```

---

## 📝 总结

**Exp 6.1b.2 = Exp 6.1b.1 + source_paper** 的意思是：

1. **Exp 6.1b.1**: 使用前后文增强 citation_context
2. **Exp 6.1b.2**: 在 6.1b.1 的基础上，再添加 source_paper 的 title 和 abstract

**source_paper 包含**:
- title: 论文标题（核心主题）
- abstract: 论文摘要（详细内容）
- categories: 论文类别
- year: 发表年份

**预期效果**: 提供更完整的语义信息，帮助模型更好地理解引用句的上下文和主题。

