---
language:
- zh
pretty_name: Wikipedia Chinese Dataset
size_categories:
- 100M<n<1B
---

# Wikipedia Chinese Dataset

中文维基百科（Wikipedia 中文版）离线数据集 [zhwiki dump](https://dumps.wikimedia.org/zhwiki)，按日期快照保存，适用于自然语言处理、信息检索、知识图谱构建等任务。

## 📦 数据集简介

本数据集包含多个时间点的中文维基百科全文快照，数据以 JSON 格式存储，每条记录包含唯一 ID、标题、标签和正文内容。  
适合用于：
- 语言模型预训练 / 微调
- 文本分类、聚类
- 知识抽取与问答系统
- 信息检索与索引构建

## 🗂 文件列表

| 文件名 | 大小 | 更新时间 |
| ------ | ---- | -------- |
| `wikipedia-zh-cn-20240901.json` | 2.12 GB | 2024-09-01 |
| `wikipedia-zh-cn-20241020.json` | 2.13 GB | 2024-10-20 |
| `wikipedia-zh-cn-20250320.json` | 2.18 GB | 2025-03-20 |
| `wikipedia-zh-cn-20250901.json` | 2.25 GB | 2025-09-01 |


## 📑 数据格式

每个 JSON 文件由多行 JSON 组成（JSON Lines 格式），每行是一条维基百科条目，包含以下字段：

| 字段名 | 类型 | 描述 |
| ------ | ---- | ---- |
| `id`    | `string` / `int` | 条目唯一标识符 |
| `title` | `string` | 维基百科条目标题 |
| `tags`  | `array[string]` | 条目标签或分类 |
| `text`  | `string` | 条目正文内容（纯文本） |

### 示例

```json
{
  "id": "123456",
  "title": "人工智能",
  "tags": ["科技", "计算机科学", "机器学习"],
  "text": "人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支..."
}
```

## 🚀 加载方法

### 使用 Hugging Face `datasets` 库

```python
from datasets import load_dataset

# 加载最新版本（2025-09-01）
dataset = load_dataset("fjcanyue/wikipedia-zh-cn", data_files="wikipedia-zh-cn-20250901.json", split="train")

# 查看样例
print(dataset[0])
```

## ⚠️ 注意事项

- 数据来源于维基百科中文站点，版权归维基百科及其贡献者所有。
- 数据体积较大，加载时请确保有足够的内存或使用流式读取。

## 📜 许可证

本数据集来源于 [Wikipedia 中文版](https://zh.wikipedia.org/)，原始文本内容遵循以下协议：

- [GNU Free Documentation License 1.3](https://www.gnu.org/licenses/fdl-1.3.html)
- [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)

除非另有说明，文本内容可在遵守上述协议的前提下自由共享、修改和再发布（包括商业用途）。  
部分内容可能仅适用 CC BY-SA 4.0 协议，或属于合理使用/其他版权例外，请在使用前确认。  
图片、媒体文件的授权信息请参考其在维基百科的描述页。

更多信息请参考 [Wikimedia Dumps License Information](https://dumps.wikimedia.org/legal.html)。

---

**作者**: [fjcanyue](https://huggingface.co/fjcanyue)  
**数据来源**: [Wikipedia 中文版](https://zh.wikipedia.org/)

