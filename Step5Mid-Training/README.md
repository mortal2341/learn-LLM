# Step5 · 中期训练 / SFT / 参数高效微调

**这个阶段解决的问题**：把 Step4 的预训练基座 `pretrain-qwen-0.1B` 变成能对话、能推理（思维链）的模型，并覆盖 LoRA / 模型融合等 PEFT 技术。

## 模型链位置

```
pretrain-qwen-0.1B (Step4)
        │
        ▼
midtrain-qwen-0.1B（本步 · ctx 512→2048）
        │
   ┌────┼─────────────┐
   ▼    ▼             ▼
sft-baike  sft-plus   R1 蒸馏分支
(120w百科) (300w指令)  (思维链)
           │
           ▼
      进入 Step6 DPO
```

## 训练成果（真实日志数据，见各目录 `log/`）

| 训练 | 模型 | 数据 | 训练 loss | 验证 loss | 训练量 |
|---|---|---|---|---|---|
| 中期训练 | `midtrain-qwen-0.1B` | 80 万条 | 均值 2.69 | 2.76 → **2.62** | 6,240 步 / 2.2 h |
| SFT · 百科 | `sft-baike-qwen-0.1B` | 120 万条多轮对话 | 2.99 → **1.11** | 1.27 → **1.14** | 4 卡 DeepSpeed |
| SFT · 指令强化 | `sft-plus-qwen-0.1B` | 300 万条中文指令 | 2.60 → **1.04** | 1.22 → **1.07** | 4 卡 DeepSpeed |
| R1 思维链蒸馏 | — | R1 蒸馏数据 | 2.38 → **1.55** | 1.76 → **1.65** | 5 epoch / 2,825 步 / 1.2 h |

四组训练的完整 log_history 均保留在 `full_sft/log/` 与 `mid_training/log/` 下，曲线可视化见 `plot.ipynb`。

## 子模块

### `mid_training/` —— 继续预训练 + 上下文扩展
- `mid_train.py`：加载预训练权重，**上下文 512 → 2048，rope_theta 10,000 → 100,000（ABF 调大基频）**，在 80 万条高质量语料上继续预训练（lr 5e-5 / warmup 2% / 1 epoch / 4 卡）
- 使用自定义 `qwen3.py` 模型实现（`qwen3.Qwen3Config` / `qwen3.Qwen3ForCausalLM`），而非直接调 transformers
- `run_mid.sh`：DeepSpeed ZeRO-2 四卡启动

### `full_sft/` —— 双分支全参 SFT
- `sft_train.py`（百科分支）：120 万条多轮对话，**逐 token loss 掩码**——system / user 轮置 `IGNORE_TOKEN_ID=-100`，只对 assistant 回复计算损失
- `sft_plus_train.py`（指令分支）：300 万条中文指令数据，作为 DPO 的上游
- `distill_r1_train.py`：R1 思维链蒸馏训练
- `run_sft.sh`：DeepSpeed 四卡启动（含 ZeRO-2 + TP 配置）
- `test_process_func.ipynb`：SFT 数据处理函数的逐 token 验证（掩码正确性）

### `lora_qlora/` —— 手写 PEFT
- `LoRA.ipynb`：**手写 LoRA 层**（`LinearLoRALayer`）：下投影 A 高斯初始化 / 上投影 B 零初始化、scale = alpha/r、冻结主干、merge/unmerge 可逆性验证（**实测误差 0.0**）
- `NF4.ipynb`：QLoRA 的 NF4 量化原理与实现（150 KB）

### `qwen32b_cot/` —— 大模型造数据
- `cot_main.ipynb`：vLLM 起 **Qwen 32B 教师服务**（`vllm_server.sh`），批量生成中文思维链数据
- 对应 Step4 `prepare_data/` 下载的 R1 蒸馏数据源

### `model_merging/` —— 模型融合
- `model_merging_main.ipynb`：模型权重融合实验（基于 Qwen2.5-Math）

### `distill_reasoning/` —— 推理蒸馏
- `distill_reasoning.ipynb`：推理能力蒸馏流程

## 关键实现细节

- **loss 掩码**：`process_func` 中对多轮对话逐段构造 `input_id` / `target`，user 段全部置 -100，assistant 段保留内容 token + `<|im_end|>`——这是 SFT 区别于预训练的核心
- **ABF 上下文扩展**：不重训 RoPE，只调大 base 提升 wavelength，配合 2048 长度语料继续预训练
- 两条 SFT 分支刻意共享上游（midtrain）、对比不同数据配方的效果
