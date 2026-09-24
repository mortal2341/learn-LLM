# Step1 · Transformer 逐组件手写

**这个阶段解决的问题**：把原版 Transformer（Attention Is All You Need）拆成最小可验证单元，每个组件单独实现、单独跑通，最后组装成完整分类模型。

## 内容清单

| 文件 | 组件 | 关键点 |
|---|---|---|
| `Attention.py` | 缩放点积注意力 | softmax 前的 scale、mask 机制 |
| `AI MultiheadAttention.py` | 多头注意力（面向面试的手写版） | QKV 投影 → 分头 → 缩放点积 → 合并 → 输出投影；附注意力权重热力图可视化 |
| `Encoder.ipynb` | 编码器 | 自注意力 + FFN + 残差 + LayerNorm 堆叠 |
| `Decoder.ipynb` | 解码器 | 掩码自注意力 + 交叉注意力 |
| `FeedForward.ipynb` | 前馈网络 | 两层线性 + 激活 |
| `Layernorm.ipynb` | 层归一化 | 逐样本归一化 vs BatchNorm |
| `Loss.ipynb` | 损失函数 | 交叉熵 + label smoothing |
| `position encoding.ipynb` | 位置编码 | 正弦位置编码（对应 Step2 的 RoPE、Step3 的 YaRN） |
| `Tokenizer.ipynb` | 分词 | 分词原理与实现 |
| `Sampling.ipynb` | 解码策略 | greedy / temperature / top-k / top-p 采样（85 KB notebook，含大量实验输出） |
| `transformer_classifier.py` | 完整模型 | 23 KB：组装以上全部组件做文本分类 |

## 学习方式

每个组件一个 notebook，单独构造输入验证形状与数值行为（例如掩码上三角 -inf 对注意力分布的影响），而不是一次性抄一个完整实现。`AI MultiheadAttention.py` 中的实现附带了 causal mask、batch 维度处理和 4 头热力图可视化，可直接用于讲解。

## 与后续 Step 的关系

- 掩码注意力 → Step2 的 KV Cache、Step3 的 MLA/DSA
- 位置编码 → Step2 的 RoPE → Step3 的 YaRN 外推
- LayerNorm → Step2 的 RMSNorm（去均值中心化）
- FFN → Step2 的 SwiGLU
