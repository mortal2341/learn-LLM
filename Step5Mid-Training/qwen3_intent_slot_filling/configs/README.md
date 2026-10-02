# DeepSpeed 配置说明（Qwen3-8B + LoRA r=8 意图/槽位联合抽取）

## 0. 先说结论：这个项目该不该开 DeepSpeed

**该开，但只该开 ZeRO-3，而且要清楚它买的是什么。**

原因：这个项目是 **LoRA r=8**，可训练参数只有 **21,823,488（21.82M）**，占 8.19B 基座的 **0.2665%**。
单卡显存的大头是**逐卡完整复制的、冻结的 bf16 基座权重**，而不是优化器状态。

| 显存项（每卡） | DDP（现状） | ZeRO-1 | ZeRO-2 | ZeRO-3 |
| --- | --- | --- | --- | --- |
| 冻结基座权重 bf16（8.19B） | 15.25 GiB | 15.25 GiB | 15.25 GiB | **3.81 GiB** |
| LoRA 权重 fp32（21.82M） | 0.081 GiB | 0.081 GiB | 0.081 GiB | 0.020 GiB |
| LoRA 梯度 fp32 | 0.081 GiB | 0.081 GiB | 0.020 GiB | 0.020 GiB |
| AdamW m + v（fp32） | 0.163 GiB | 0.041 GiB | 0.041 GiB | 0.041 GiB |
| **小计（不含激活）** | **15.58 GiB** | 15.45 GiB | 15.39 GiB | **3.90 GiB** |
| 相比 DDP 省下 | — | **−0.12 GiB** | **−0.18 GiB** | **−11.68 GiB** |

结论：

- **ZeRO-1 / ZeRO-2 在这个项目里几乎没有意义**：LoRA 的优化器状态和梯度加起来才 0.24 GiB，4 卡分片只能省 0.2 GiB 左右，白付通信开销。
- **ZeRO-3 是唯一有实质收益的**：它把冻结的基座权重也切开，单卡从 15.25 GiB 降到 3.81 GiB（4 卡），换来约 **11.7 GiB/卡** 的空闲显存，可用来放大 batch、加长序列、或换更大的基座（Qwen3-14B / 32B）。

作为对照，如果这是**全参微调** 8.19B（AdamW 混合精度）：权重 15.25 + 梯度 15.25 + fp32 master 30.5 + m 30.5 + v 30.5 ≈ **122 GiB**，那时 ZeRO-2 能把每卡压到约 34 GiB、ZeRO-3 压到约 30.5 GiB —— 那才是 ZeRO 的主场。**LoRA 场景下 ZeRO 收益小，就是因为可训练状态本来就小。**

## 1. 文件清单与怎么用

| 文件 | 用途 |
| --- | --- |
| `configs/ds_zero2.json` | ZeRO-2（分片优化器状态 + 梯度）。留作对照 / 全参微调时使用 |
| `configs/ds_zero3.json` | **推荐**。ZeRO-3，分片全部参数 |
| `configs/ds_zero3_offload_optim.json` | ZeRO-3 + 优化器状态卸载到 CPU。本项目收益极小，仅备用 |
| `train_ds_zero2.sh` | 用 ZeRO-2 启动 4 卡训练 |
| `train_ds_zero3.sh` | 用 ZeRO-3 启动 4 卡训练 |
| `train_ner.py:127` | 新增 `deepspeed=os.environ.get("DS_CONFIG") or None` |

启动方式：

```bash
# ZeRO-3（推荐）
bash train_ds_zero3.sh

# ZeRO-2
bash train_ds_zero2.sh

# 不设置 DS_CONFIG 时，行为与改动前完全一致（纯 DDP），老脚本 train_ds.sh 照旧可用
bash train_ds.sh
```

`train_ner.py` 的改动是**可选开关式**的，不设环境变量即 `deepspeed=None`，不会影响已有流程。

> DeepSpeed 已经装好了（`output/multi_log.txt:3` 里就是 `python -m deepspeed.launcher.launch`），不需要额外安装。

## 2. 每个字段优化了什么

### 2.1 精度

| 字段 | 作用 |
| --- | --- |
| `bf16.enabled: true` | 用 bfloat16 做前向/反向与通信。相比 fp32 权重和激活直接减半；相比 fp16 动态范围大（8 位指数），**不需要 loss scaling，不易溢出**，这是它替代 fp16 的主要原因。必须和 `TrainingArguments` 里加载模型时的 `dtype=torch.bfloat16` 一致，否则会出现 dtype 冲突报错 |
| `fp16.enabled: false` | 显式关掉，避免和 bf16 同时开启 |

### 2.2 ZeRO 分片（核心）

ZeRO 把训练状态分三级切开，一级比一级省得多：

| 字段 | 优化了什么 |
| --- | --- |
| `stage: 2` / `stage: 3` | 分片对象。1=优化器状态，2=+梯度，3=+模型参数 |
| `offload_optimizer.device: cpu` | 把优化器状态（m/v）放到 CPU 内存。本项目只有 0.163 GiB，**没必要** |
| `offload_param.device: none` | 保持参数在 GPU。**特别注意**：ZeRO-3 如果开 `offload_param`，每次前向都要通过 PCIe 把 15 GiB 冻结权重拉回 GPU，会慢到不可用，所以这里坚决设 `none` |
| `overlap_comm: true` | 让 all-gather / reduce-scatter 通信与反向计算重叠，**用显存换速度**（会多占一点 bucket 显存） |
| `contiguous_gradients: true` | 梯度写入一块连续缓冲区，减少显存碎片、合并小 all-reduce 调用，降低通信次数 |
| `sub_group_size: 1e9` | ZeRO-3 中参数更新的分组粒度，只在配合 CPU offload 时才需要调小；不 offload 就设大值避免频繁换入换出 |
| `stage3_gather_16bit_weights_on_model_save: true` | **保存时把分片权重聚合回完整 bf16 模型**。ZeRO-3 不设这个，`save_pretrained` 出来的东西不能用 |
| `stage3_prefetch_bucket_size` | 预取参数桶大小：调大→显存换延迟，调小→延迟换显存 |
| `stage3_param_persistence_threshold` | 小于该值的参数**不分片、常驻显存**，减少 all-gather 次数（LoRA 的 A/B 矩阵很小，受益于此） |
| `stage3_max_live_parameters` | 同时驻留在显存里的参数量上限，控制峰值显存 |
| `stage3_max_reuse_distance` | 参数重用窗口，窗口内不释放，减少反复 gather |

### 2.3 通信桶（省显存 vs 省带宽的旋钮）

| 字段 | 作用 |
| --- | --- |
| `reduce_bucket_size: 2e8` | reduce-scatter 的桶大小。调大→通信次数少但峰值显存高。默认 5e8，4 卡 8B 场景下调到 2e8 更稳 |
| `allgather_bucket_size: 2e8` | ZeRO-1/2 里 all-gather 分片的桶大小，同上权衡 |
| `reduce_scatter: true` | 用 reduce-scatter 替代 all-reduce，省通信量 |
| `allgather_partitions: true` | 前向时按需 all-gather 参数分片 |
| `round_robin_gradients: true` | （ZeRO-2）把梯度按轮转顺序分桶，缓解不同 rank 负载不均导致的等待 |

### 2.4 与 HuggingFace Trainer 的衔接（`"auto"` 的坑）

写 `"auto"` 的字段会被 `Trainer` 按 `TrainingArguments` 自动填：

| 字段 | 自动填成什么 |
| --- | --- |
| `train_batch_size: "auto"` | `4 × 8 × 4 = 128` |
| `train_micro_batch_size_per_gpu: "auto"` | `per_device_train_batch_size = 4` |
| `gradient_accumulation_steps: "auto"` | `8` |
| `gradient_clipping: "auto"` | `max_grad_norm = 1.0` |
| `reduce_bucket_size` / `stage3_prefetch_bucket_size` / `stage3_param_persistence_threshold: "auto"` | 按模型 `hidden_size=4096` 推导 |

**重要：JSON 里不要写 `optimizer` 和 `scheduler` 段。** 一旦写了，DeepSpeed 会接管优化器/调度器，覆盖掉 `TrainingArguments` 里的 `optim="adamw_torch"` 和 `lr_scheduler_type="cosine"` + `warmup_ratio=0.01`，学习率曲线就和你实验结果对不上了。

## 3. 开了 ZeRO-3 之后，代价是什么

### 3.1 速度

ZeRO-3 每一层前向/反向都要 all-gather 参数，**包括冻结的基座权重**。
DDP 的通信只在反向结束后一次 all-reduce；ZeRO-3 是每层多次通信，所以：

- 你的 **DDP 基线**（`multi_log.txt`）：`train_steps_per_second = 0.398`，1668 步 / 4195.8 s = **69.9 分钟**。
- 开 ZeRO-3 后请直接对比这个数，🔎 **经验预期慢 20%~40%**（`overlap_comm: true` 能捞回一部分）。

也就是说：**如果 4 卡已经跑得下，开 ZeRO-3 是拿速度换显存。** 只有在"显存不够 / 想放大 batch 或换大模型"时才划算。

### 3.2 保存与合并链路

- `stage3_gather_16bit_weights_on_model_save: true` 已开，`trainer.save_model()` 会聚合出完整权重。
- **不要**在 ZeRO-3 的多进程环境里直接跑 `merge_and_unload()`。
- 正常做法：训练结束后在**单卡**上按原逻辑合并（`merge_model.py` 加载 `Qwen/Qwen3-8B` + 新 checkpoint 的 adapter），流程不用改。
- PEFT 的 `save_pretrained` 对 ZeRO-3 有 `GatheredParameters` 处理，但保存后建议校验 adapter 不是全零。

### 3.3 评估

`eval_strategy="steps", eval_steps=100` 在 ZeRO-3 下每次评估都要 gather 参数，开销变大。
建议一起改成 `eval_steps=500`，和原来的粒度对齐。

## 4. 更划算的替代方案（按性价比排序）

| 优化 | 效果 | 代价 |
| --- | --- | --- |
| `gradient_checkpointing=True` | 激活显存砍掉**大部分**（通常几 GB），是 LoRA 训练最立竿见影的一项 | 慢约 20%~30%；PEFT 下需注意 `enable_input_require_grads` |
| **QLoRA（4bit 基座 + LoRA）** | 基座 15.25 GiB → **约 4.1 GiB**，比 ZeRO-3 还省，且**无额外通信**，单卡就能训 | 需 `bitsandbytes`；4bit 量化有轻微精度损失；推理前要 `merge_and_unload` 回 bf16 |
| ZeRO-3 | 每卡省约 11.7 GiB | 通信变多，慢 20%~40%，保存/合并变复杂 |
| `optim="adamw_bnb_8bit"` | 优化器状态 0.163 → 0.041 GiB | 本项目收益可忽略，不推荐单独使用 |
| FlashAttention / SDPA | 降低注意力激活显存 + 提速 | 需装 flash-attn，且要匹配 CUDA 版本 |
| 动态 padding / packing | 减少无效 token，提升有效吞吐 | 要改数据管线 |
| `dataloader_num_workers` 调大 | 解决数据加载瓶颈 | 与你 4 卡 GPU 利用率相关 |

**如果要真正省显存，建议的组合是：`gradient_checkpointing=True` + ZeRO-3**，或者直接上 **QLoRA**。

## 5. 验证清单（跑起来之后逐项确认）

1. 启动日志里出现 `DeepSpeed ZeRO Stage3`，`world_size=4`。
2. `Trainer` 打印的 `train_batch_size` 等于 128（说明 `"auto"` 生效）。
3. `nvidia-smi` 单卡显存从 DDP 时的水平下降到 4~6 GiB 权重区间 + 激活（🔎 预期下降 10 GiB 以上）。
4. 对比 `train_steps_per_second` 与基线 `0.398`，量化速度损失。
5. 训练结束检查 `checkpoint-*/adapter_model.safetensors` 大小是否仍约 83 MiB、数值非全零。
6. 单卡跑通 `merge_model.py`，再用 `predict_vllm.py` 复算联合准确率，与基线 **0.9448**（4492 条）对比，确认精度没有掉。
