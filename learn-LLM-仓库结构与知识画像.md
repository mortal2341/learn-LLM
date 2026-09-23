# learn-LLM 仓库结构与知识画像诊断

> 仓库：`mortal2341/learn-LLM`（公开）
> 描述：记录个人的大模型学习之旅，以及项目代码
> 快照时间：2026-09-23
> 结论提要：这是一条**从 PyTorch 算子手写 → Transformer 拆解 → 数据工程 → 0.1B 预训练 → SFT/蒸馏 → DPO/PPO/GRPO 自研分布式 RL 框架**的完整闭环，覆盖大模型全生命周期 8 个阶段，深度显著超过「调包型」水平。

---

## 一、仓库概览

| 项目 | 数值 |
|---|---|
| 提交数 | 3（均为 2026-09-23） |
| 已上传文件 | 41,683 个 / 约 1.09 GB |
| 未上传（本地保留） | 196 个 / 约 108.74 GB（≥10MB 数据集与权重） |
| 导入来源 | 本地 `D:\learn-torch`，严格保留原始目录层级 |
| 根目录 | `.claude/`、`.vscode/`、`.gitignore`(20 KB)、`README.md`(70 B) + 10 个内容目录 |

**Git 操作质量说明**：使用了 `--allow-unrelated-histories` 合并远端初始提交与本地导入，并明确声明「未使用强制推送覆盖」；`.gitignore` 中按体积逐条列出排除清单。这一操作规范程度高于大多数个人学习仓库。

---

## 二、目录结构逐层拆解

### foundation/ —— 底层复现
手写 PyTorch 基础组件：`nn,module.py` / `nn,seq.py` / `nn.conv.py` / `nn.conv2d.py` / `nn,maxpool.py` / `nn.relu.py` / `read_data.py` / `dataset`（CIFAR-10）。
→ 意味着理解了 `nn.Module` 的前向/反向契约、卷积的滑窗与 im2col、池化的下采样逻辑，而不是只会 `nn.Conv2d(...)`。

### Step0 pytorch —— 训练循环与序列建模入门
- `autograd.py`：手写梯度下降 `w -= lr * w.grad` + TensorBoard 记录（不依赖 optimizer）
- `grad.py` / `grad_pytorch.py`：梯度计算对照实验
- `CatVSDog.py` / `fenlei shoudong.py`：手写分类流程
- `RNN translator/`：中英翻译器，含**自训的 `zh_bpe.model` / `en_bpe.model` 词表**与 `translator.py`（14 KB）

### Step1 Transformer —— 全组件手写
`Attention.py`、`AI MultiheadAttention.py`（含注意力热力图可视化）、`Encoder.ipynb`、`Decoder.ipynb`、`FeedForward.ipynb`、`Layernorm.ipynb`、`Loss.ipynb`、`Sampling.ipynb`、`Tokenizer.ipynb`、`position encoding.ipynb`、`transformer_classifier.py`（23 KB 完整分类模型）。
→ 注意力、残差、LayerNorm、交叉熵、采样策略、位置编码、分词 —— 每一项单独成篇，属于**逐块验证式**的学习方式。

### Step2 LLaMA —— 现代 LLM 结构件
`RMSNorm.ipynb`、`FFN_SwiGLU.ipynb`、`ROPE.ipynb`、`GQA KVcache.ipynb`。
→ 正好是 LLaMA 相对原版 Transformer 的四个关键改动，说明是按「论文差异点」逐个攻克的。

### Step3 DeepSeek —— 架构深度拆解（仓库最硬核的部分）
| 文件 | 对应技术 |
|---|---|
| `MLA_Prefill.ipynb` / `MLA_Decoding.ipynb` / `MLA_Decoding V2.ipynb` / `mla_cache.npy` | 多头潜在注意力，**按 Prefill / Decoding 两阶段分别实现并做缓存** |
| `DSA-Prefill.ipynb` | DeepSeek Sparse Attention |
| `DeepSeek_MOE.ipynb` / `MOE_LoadBalance.ipynb` | MoE 路由 + 负载均衡损失 |
| `MTP.ipynb` | Multi-Token Prediction 多 token 预测 |
| `YaRN.ipynb` | 长上下文位置插值外推 |
| `DeepSeek-Distill-Qwen.ipynb` | 蒸馏链路 |
| `ROPE.py` | 独立 RoPE 实现（含 cos/sin cache 动态扩展） |

→ 这不是「读论文」，而是**把 DeepSeek-V3 技术报告的核心模块逐个落地成代码**，属于国产大模型原理层面第一梯队的理解深度。

### Step4pretrain —— 预训练全流程（数据工程 + 训练 + 评测）
- `custom_tokenizer/`：`train_tokenizer.py` 自训 **BBPE 词表（vocab 9600）**，特殊 token 完整设计（`<|im_start|>`、`<think>`、`<answer>`、`<tool_call>`、`<|endoftext|>` 等）；`merge_tokenizer.py` 词表合并；`stats.py` 做字符分布统计
- `prepare_data/`：`download.sh` 拉取 wikibaike / firefly / belle / Infinity-Instruct / R1-Distill 多源语料；`format_raw_data*.py` 四个脚本分别处理 **512 / 2048 两种长度**，并分流为预训练 / SFT / 思维链三类
- `quality_model/`：用 `bert-base-chinese` 训练**数据质量分类器**；`deduplicate.ipynb` + **手写 MinHash LSH 去重**（`minhash LSH.py`）
- `data_js/`：Data-Juicer 配置 + `log_2048.txt`（288 KB **真实运行日志**）
- `train_llm/`：`pre_train.py` 用 Qwen3 架构**从零配置 0.1B 模型**（16 层 / hidden 512 / 8 Q heads / 4 KV heads / rope_theta 10000 / vocab 58703）在 230 万条数据上训 2 epoch；`continue_train.py`、`cal_ppl.py`、`generate.py`、`lm-evaluation-harness` + `test_ceval.sh` 做中文评测

→ **数据工程是这条链路里最稀缺的能力**，这里从分词器、质量筛选、去重、多源格式化到评测形成闭环。

### Step5Mid-Training —— 中期训练 / SFT / 参数高效微调
- `mid_training/`：`mid_train.py`（4 卡 DeepSpeed ZeRO-2）、`run_mid.sh`、数据清洗、PPL 计算
- `full_sft/`：`sft_train.py`、`sft_plus_train.py`、`distill_r1_train.py`（**R1 思维链蒸馏**）、4 卡 DeepSpeed 启动脚本
- `lora_qlora/`：**手写 LoRA 层**（`LinearLoRALayer`，B 零初始化 / A 正态初始化 / scale=alpha/r / merge & unmerge 一致性验证，误差 0.0）、`NF4.ipynb` 量化
- `model_merging/`：模型融合 + Qwen2.5-Math 实验
- `qwen32b_cot/`：vLLM 起服务，用 **32B 教师模型批量造 CoT 数据**

### Step6 Reinforcement Learnig —— 强化学习地基 + 对齐实战
- `rl-base/`：**从零手写整套 RL 算法** —— `REINFORCE.ipynb`、`ActorCritic.ipynb`、`DQN.ipynb`、`MonteCarlo.ipynb`、`TrainMonteCarlo.ipynb`、`TD.ipynb`、`PPO.ipynb` + `DPO_loss.py`（手写 DPO loss）+ `utils.py`
- `dpo/`：`dpo.ipynb`、`orpo.ipynb`、`apo.ipynb` 三种偏好优化
- `dpo_train/`：`dpo_train.py` + `dpo_evaluate.py` + `preprocess.sh` + 完整绘图
- `ad_text_gen/`：**真实业务项目** —— 基于 Phi-4 + LlamaFactory 的电商服装广告文案生成；`prompts.py` 是精心设计的结构化 prompt；`generate_chosen_data.ipynb` / `generate_rejected_data.ipynb` **自造偏好对**；`cli.py` 做 LoRA 推理

→ 「先手写 PPO，再用 DPO，最后落到业务」的路径，比直接调 TRL 扎实得多。

### Step7RL —— 自研分布式 RL 训练框架（最强项）
`7.post_train-stage/rlhf_grpo/train.py`（22 KB，**完全自研**）：

| 能力点 | 具体实现 |
|---|---|
| 三种算法 | `GRPO_step` / `DAPO_step` / `GSPO_step` 分别实现，含 K3 KL 估计 |
| DAPO 细节 | Clip-Higher（上下 clip 不对称）、**去掉 KL 约束**、Token-Level Policy Gradient Loss |
| GSPO 细节 | **序列级重要性采样**（组内 logp 均值）、序列级 clip |
| 架构解耦 | vLLM 独立进程做 rollout，DeepSpeed 进程做训练，`mp.Queue` 同步权重 |
| 参考模型服务 | `ref_server.py` 用 bottle + tornado **自建 HTTP 服务算 ref logp**，二进制 bytes 协议通信，LIFO 队列 + tensor/string 模式轮换 |
| GRPO 要点 | 组内优势归一化 `(r-mean)/(std+eps)`、**无区分度组自动丢弃**、fp32 保精度 |
| 显存优化 | 循环算 log_softmax 降低峰值显存、pad_sequence 打包 |
| 评测闭环 | 格式准确率 + 答案准确率双指标，实时评估 |
| badcase 分析 | rollout 结果写 `reasoning_log_*.txt` |

同目录另有：`on_policy_distill/`（18 KB，on-policy 蒸馏 + 教师服务）、`rlhf-ppo/`（RLHF-PPO notebook）、`grpo_medical/`（GRPO + Unsloth LoRA，医疗领域，含单卡 32G 版本）、`r_zero/`（接 verl 框架 + 自研题库生成/打分）、`train_gold/`。

→ **能自己搭出一套 vLLM rollout + 参考模型 server + DeepSpeed 训练解耦的 RL 训练框架，这是绝大多数写「熟悉 RLHF」的人做不到的事。**

---




---

*本报告基于 GitHub 仓库 `mortal2341/learn-LLM` 在 2026-09-23 的文件树与源码内容生成。评分是依据代码证据密度的主观评估，用于定位能力分布，不等同于任何官方认证。*
