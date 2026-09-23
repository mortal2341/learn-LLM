---
configs:
- config_name: code
  default: true
  data_files:
  - split: train
    path: "code/*.parquet"
- config_name: math
  data_files:
  - split: train
    path: "math/*"
- config_name: general
  data_files:
  - split: train
    path: "general/*"
- config_name: math(reasoning)
  data_files:
  - split: train
    path: "math(reasoning)/*"
- config_name: code(reasoning)
  data_files:
  - split: train
    path: "code(reasoning)/*"
license: cc-by-nc-sa-4.0
task_categories:
- question-answering
- text-generation
- text2text-generation
language:
- zh
size_categories:
- 10M<n<100M
pretty_name: Chinese-Instruct-Lite
---

<h1 align="center">
  中文指令微调数据集 - Lite 版本
</h1>

<p align="center">
  <a href="https://github.com/Mxoder/Maxs-Awesome-Datasets" target="_blank">💻 Github Repo</a> <br>
</p>

---

> [!TIP]
> 这不是 [Chinese-Instruct](https://huggingface.co/datasets/Mxode/Chinese-Instruct) 的子集，而是一个**全新**的简化数据集。
>
> 如果您想要一个可以真实使用、而不仅仅适用于学习的数据集，欢迎访问：[Mxode/Chinese-Instruct](https://huggingface.co/datasets/Mxode/Chinese-Instruct)
>
> 如果您想要一个更加简单易收敛、主题集中的数据集，可以访问：[Mxode/I_Wonder_Why-Chinese](https://huggingface.co/datasets/Mxode/I_Wonder_Why-Chinese)


## 具体构成

本数据集包含如下 5 个子集，总数据量 10M+。

- `code`：**代码**主题的指令数据集，数据量 1.2M+。

- `math`：**数学**主题的指令数据集，数据量 1.7M+。

- `general`：通用指令数据集，主题广泛，与 `code` 和 `math` 指令不重复，数据量 5.1M+。

- `math(reasoning)`：**数学**推理数据集，指令采样自 `math` 子集，可通过 `id` 关联，数据量 1.2M+。

- `code(reasoning)`：**代码**推理数据集，指令采样自 `code` 子集，可通过 `id` 关联，数据量 700K+。


## 如何使用

你应当明确指定需要加载的子集，例如：

```python
from datasets import load_dataset

# 指定加载 code 子集
ds = load_dataset("Mxode/Chinese-Instruct-Lite", "code")
```

## 适用范围

这个数据集的初始构造目的，就是为了方便学习性质的从零训练中文大模型类（LLM from scratch）项目，因此难度和广度是有上限的。
比较推荐用于 **< 2B、特别是 < 0.5B 模型**的训练，不推荐将这个数据集用于**经过良好中文训练的模型**。



## 局限性

Chinese-Instruct-Lite 是一个合成数据集，它的核心价值在于开放性、包容性和助人精神。它旨在服务于所有人，并适用于广泛的应用场景。
请注意，尽管 Chinese-Instruct-Lite 尽力进行了过滤，但仍然难以保证最终内容是完全准确、无偏见的。
因此，在使用 Chinese-Instruct-Lite 前应当根据使用场景进行更加详尽的选择和过滤。