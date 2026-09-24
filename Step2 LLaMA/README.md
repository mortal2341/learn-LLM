# Step2 · LLaMA 结构拆解

**这个阶段解决的问题**：LLaMA 相对原版 Transformer 改了哪四个地方？逐一实现并理解每个改动的动机。

## 内容清单

| 文件 | 组件 | 相对原版 Transformer 的改动 |
|---|---|---|
| `RMSNorm.ipynb` | RMSNorm | 去掉均值中心化，只做缩放归一化；比 LayerNorm 少一次统计计算 |
| `FFN_SwiGLU.ipynb` | SwiGLU 前馈 | 门控激活（SiLU）替代 ReLU/GLU，三个投影矩阵（gate / up / down） |
| `ROPE.ipynb` | 旋转位置编码 | 用复数旋转替代正弦绝对位置编码，注意力内积天然带相对位置信息（234 KB notebook） |
| `GQA KVcache.ipynb` | GQA + KV Cache | 多查询分组：KV 头数 < Q 头数，推理时 KV Cache 显存大幅下降（35 KB） |

## 为什么按这四个组件学

这四项正好是 LLaMA 系（也是当前几乎所有主流开源模型：Qwen / DeepSeek / Mistral）的标准结构。逐个实现之后，阅读任何现代 LLM 的模型代码时，剩下的差异只有：

- 注意力的变体（Step3 的 MLA / DSA）
- FFN 的稀疏化（Step3 的 MoE）
- 位置编码的外推策略（Step3 的 YaRN）

也就是说，**Step2 + Step3 覆盖了从原版 Transformer 到 DeepSeek-V3 的全部结构差异**。

## 关键实现细节

- RoPE：`rotate_half` 的实现方式、cos/sin 缓存、按位置索引取用（Step3 的 `ROPE.py` 保留了独立可复用版本，含缓存动态扩展）
- GQA：KV 头如何在组内重复利用，Cache 命中率与显存的权衡
