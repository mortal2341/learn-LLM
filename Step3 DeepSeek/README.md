# Step3 · DeepSeek 架构代码级复现

**这个阶段解决的问题**：把 DeepSeek-V3 技术报告里的核心模块逐个落地成可运行的代码。这是全仓库原理侧最硬核的部分。

## 内容清单

| 文件 | 模块 | 说明 |
|---|---|---|
| `MLA_Prefill.ipynb` | 多头潜在注意力 · Prefill | 压缩 KV 到潜在空间，预填充阶段实现（895 KB） |
| `MLA_Decoding.ipynb` | MLA · Decoding | 解码阶段的吸收式计算 |
| `MLA_Decoding V2.ipynb` | MLA · Decoding 优化版 | 对比两种实现路径的效率 |
| `mla_cache.npy` | MLA 缓存产物 | 实际运行后落盘的潜在 KV 缓存 |
| `DSA-Prefill.ipynb` | DeepSeek 稀疏注意力 | 稀疏注意力在预填充阶段的实现 |
| `DeepSeek_MOE.ipynb` | MoE 路由 | 专家路由与前向 |
| `MOE_LoadBalance.ipynb` | MoE 负载均衡 | 辅助负载均衡损失，防止专家塌缩 |
| `MTP.ipynb` | 多 Token 预测 | DeepSeek 的 MTP 模块：一次预测多个未来 token |
| `YaRN.ipynb` | 长度外推 | NTK-aware 插值 + 温度缩放，547 KB 实验记录 |
| `DeepSeek-Distill-Qwen.ipynb` | 蒸馏链路 | DeepSeek → Qwen 的蒸馏实验（2 MB） |
| `ROPE.py` | RoPE 独立实现 | 含 cos/sin 缓存与超长序列动态扩展，可直接复用 |

## 为什么拆 Prefill / Decoding 两阶段

MLA 的核心是「训练时压缩、推理时恢复」，而 Prefill 和 Decoding 的最优计算路径不同：

- **Prefill**：可以承受矩阵吸收（W_UK / W_UV 先乘），整段并行
- **Decoding**：逐 token 生成，重点是潜在 KV 缓存的命中与低显存

两份实现 + 缓存产物（`mla_cache.npy`）说明这不是纸面推导，而是真的把两条路径都跑通了。

## 与其他 Step 的关系

- RoPE（Step2）→ YaRN 外推（本步）
- 注意力（Step1/2）→ MLA / DSA（本步）
- FFN（Step1/2）→ MoE 稀疏化（本步）
- 本步的蒸馏链路 → Step5 的 R1 思维链蒸馏
