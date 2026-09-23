---
configs:
- config_name: general
  default: true
  data_files:
  - split: train
    path: general/*
- config_name: reasoning
  data_files:
  - split: train
    path: reasoning/*
- config_name: preference
  data_files:
  - split: train
    path: preference/*
license: cc-by-sa-4.0
task_categories:
- text-generation
- question-answering
language:
- zh
pretty_name: i-wonder-why
size_categories:
- 1M<n<10M
tags:
- chemistry
- biology
- legal
- finance
- music
- art
- climate
- medical
- synthetic
---

<h1 align="center">
  🧐 十万个为什么 - 中文百科开放问答数据集
</h1>

<p align="center">
  <a href="https://github.com/Mxoder/Maxs-Awesome-Datasets" target="_blank">💻 Github Repo</a> <br>
</p>

---

这是一个中文百科开放问答数据集，共分为 3 个子集：`general`、`preference`、`reasoning`。这个数据集可适用于 SFT 指令微调、DPO 类强化学习、R1 类推理蒸馏任务。


> [!tip]
> [2025/05/09] 发布了一个新的中文指令数据集 [Chinese-Instruct-Lite](https://huggingface.co/datasets/Mxode/Chinese-Instruct-Lite)，包含代码、数学、通用多场景，同样包含一般指令微调数据与推理数据，数据总量 10M+
>
> [2025/05/05] 更新：数据集扩增，现在指令由 600K+ 增加到 1.2M+ 了！


## 数据集详情

所有的子集共享相同的**指令（prompt）**，共计 1.2M+，每一条指令都有自己独有的 12 位 id。这意味着你可以根据 `id` 交叉混合使用不同的子集。

由于指令相同，因此所有子集的数据量都是一致的，均为 1.2M+。

- `general`：这个子集适用于 SFT 指令微调，形式是最简单的 `prompt-response` 格式。

- `preference`：偏好数据集，适用于 DPO 类强化学习，每条 `chosen` 回复对应着 3 条 `rejected` 回复，这意味着你可以构造 3 倍量的偏好数据对。当然，你也可以任意抽取 3 条 `rejected` 中之一。需要注意的是，`rejected` 内部是没有偏好序的，不能在内部混搭。

- `reasoning`：推理数据集，适用于 o1、R1 类推理任务，是经典的 `prompt-reasonging-response` 格式。由于指令的 `id` 唯一性，你也可以将这个子集与 `general` 混合组成新的偏好数据集。


## 可能的使用方法

1. 你可以直接使用 `general` 子集，进行最基本的 SFT 指令微调，可以用全参训练或者如 LoRA 等 PEFT 方法训练。

2. 你可以对 `preference` 数据集进行分割，从 3 条 `rejected` 中任意抽选或者全部抽选，最后构成 `prompt-chosen-rejected` 的三元组数据集，即常见的偏好数据集格式。然后你可以使用 DPO 等强化学习方式进行训练。如同已有实践证明的那样，DPO 前最好使用同领域的数据进行 SFT、保证分布贴近，好在本数据集都是同领域的！你可以先使用它们来微调。

3. 你可以使用 `reasoning` 数据集来训练你的模型，让其拥有像 o1、R1 那样的思维链输出能力。

4. 你可以利用 `id`，将 `general`/`preference` 中的回复和 `reasoning` 的回复拼接起来，形成一个新的偏好数据集，试验一下用传统 DPO 的方式让模型学会思考。

5. 你可能只想最简单、开销最小地训练一个 LLM，那么你可以利用 `preference` 子集中的 `rejected` 回复数据来用于 SFT 训练。不用担心，`preference` 子集中的 `rejected` 数据并不意味着质量差，只是特意构造成短回复、简略的形式，因此恰好可以用来训练一个最简单的小模型。

6. 你当然也可以抽取这些数据混入你的小模型的预训练阶段，这在实践中被证明是有利于提升下游性能的，会更容易看到最终的训练效果。

当然，以上只是一些使用方法举例，由于三个子集指令的一致对应，你完全可以自由发挥！


## 如何使用

你应当明确指定需要加载的子集，例如：

```python
from datasets import load_dataset

# 指定加载 general 子集
ds = load_dataset("Mxode/I_Wonder_Why-Chinese", "general")
```


## 适用范围

这个数据集的初始构造目的，就是为了方便学习性质的从零训练中文大模型类（LLM from scratch）项目，因此难度和广度是有上限的。比较推荐用于 < 2B、特别是 < 0.5B 模型的训练，非常不推荐将这个数据集用于**超过 7~8B 规模**的模型的训练。



## 局限性

I_Wonder_Why-Chinese 是一个合成数据集，它的核心价值在于开放性、包容性和助人精神。它旨在服务于所有人，并适用于广泛的应用场景。请注意，尽管 I_Wonder_Why-Chinese 尽力进行了过滤，但仍然难以保证最终内容是完全准确、无偏见的。因此，在使用 I_Wonder_Why-Chinese 前应当根据使用场景进行更加详尽的选择和过滤。