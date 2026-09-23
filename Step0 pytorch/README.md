# Step0 · PyTorch 训练基础与序列建模

**这个阶段解决的问题**：掌握自动求导机制与手写训练循环，并用 RNN + 自训 BPE 完成第一个序列到序列任务（中英翻译）。

## 内容清单

| 文件 | 内容 |
|---|---|
| `autograd.py` | 手写训练循环：`w -= lr * w.grad` 手动更新参数 + TensorBoard 记录 loss（不依赖 optimizer） |
| `grad.py` / `grad_pytorch.py` | 梯度计算对照实验 |
| `CatVSDog.py` | 猫狗分类完整流程 |
| `fenlei shoudong.py` | 手写分类训练（不使用 Trainer） |
| `RNN translator/` | 中英翻译器：`translator.py`（14 KB seq2seq 实现）+ 自训词表 |

## RNN 翻译器细节

`RNN translator/` 下保留了完整的双语词表产物：

- `zh_bpe.model` / `zh_bpe.vocab`（约 483 KB / 257 KB）—— 中文 BPE 词表
- `en_bpe.model` / `en_bpe.vocab`（约 507 KB / 240 KB）—— 英文 BPE 词表

这是整个仓库数据工程的起点：**词表是自己训练的**，不是拿现成的。这一步的经验直接延续到 Step4 的 BBPE 自训词表（9,600 → 合并 58,703）。

## 关键收获

- 理解 `requires_grad` / `.grad` / `zero_grad()` 的完整链路，为后续手写 LoRA、GRPO 的显式梯度操作打底
- 序列建模（encoder-decoder）的第一手经验，对照 Step1 的 Transformer 理解「为什么需要注意力」
