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

## 三、知识画像：这个仓库说明了什么

### 3.1 十维能力评估（基于代码证据密度，10 分制）

| 维度 | 评分 | 关键证据 |
|---|---|---|
| Transformer 原理与手写 | 9.0 | 全组件独立实现 + 完整分类模型 |
| PyTorch / 自动求导 / 基础网络 | 8.5 | 手写梯度下降、nn.Module/Conv/MaxPool、RNN 翻译器 |
| 现代 LLM 结构（RoPE/GQA/SwiGLU/RMSNorm） | 8.5 | 按 LLaMA 差异点逐个攻克 |
| DeepSeek 架构（MLA/MoE/MTP/DSA/YaRN） | 8.5 | MLA 分 Prefill/Decoding 实现，技术报告级拆解 |
| 强化学习与对齐 | 8.5 | 手写 REINFORCE→AC→DQN→PPO，再到 DPO/GRPO 自研 |
| SFT / 蒸馏 / LoRA | 8.5 | 全参 SFT + R1 蒸馏 + 手写 LoRA 层 + 32B 造 CoT |
| 数据工程与分词器 | 8.0 | BBPE 自训、MinHash 去重、BERT 质量分类、多源清洗 |
| 分布式训练与推理工程 | 8.0 | DeepSpeed ZeRO-2 4 卡、vLLM rollout、自建 ref server |
| 预训练与评测闭环 | 7.5 | 0.1B 从零训练 + PPL + CEval 评测 |
| 工程规范 / 文档 / 可复现性 | 5.5 | 见第五节短板 |

### 3.2 一句话画像

> **「手写优先 + 全链路无断层 + 有真实落地」的 LLM 算法工程师画像。**

三个特征最能说明问题：

1. **手写优先**：LoRA 层、MHA、MinHash LSH、Conv2d、MaxPool、PPO、GRPO loss —— 都用「自己实现 + 单元验证」的方式学，而不是调用现成 API。这决定了知识是「可推导的」而非「可复述的」。
2. **全链路无断层**：从 `nn.Conv2d` 到 `GRPO` 的剪裁目标函数，中间没有任何一层是靠调库跳过去的，这在个人学习仓库里非常少见。
3. **有真实落地与评测**：不是只跑通 demo —— 电商广告文案项目做了「SFT → 自造偏好对 → DPO → 合并推理」完整闭环；预训练接 CEval；RL 阶段接双指标评测。

### 3.3 适合定位的岗位方向
- 大模型算法工程师 / 后训练（Post-Training）工程师 —— **最匹配**
- 预训练数据工程师 —— 数据工程部分证据充分
- LLM 推理/训练框架工程师 —— 有自研分布式 RL 框架经验，但缺推理侧工程（见短板）
- 模型结构研究（MLA/MoE 方向）—— 对国产模型架构理解深入

---

## 四、亮点（建议在简历/面试中重点讲的三件事）

1. **自研分布式 GRPO/DAPO/GSPO 训练框架**
   rollout（vLLM 独立进程）与训练（DeepSpeed）解耦 + 自建参考模型 server + 二进制通信协议 + 组内优势归一化 + 无区分度组过滤。这一个项目就能撑起「后训练工程能力」的整段面试。

2. **从零预训练 0.1B 中文模型并完成评测闭环**
   自训 BBPE 词表 → MinHash 去重 → BERT 质量分类 → 230 万条数据 → Qwen3 架构 16 层模型 → PPL + CEval。数据侧和训练侧都是自己写的。

3. **DeepSeek-V3 核心模块的代码级复现**
   MLA 分 Prefill/Decoding 两阶段并做 KV 缓存、DSA 稀疏注意力、MoE 负载均衡、MTP、YaRN。说明能读透最前沿的中文技术报告并落地。

---

## 五、短板与改进建议（按优先级）

### P0 — 文档与可见性（当前最影响发挥的一环）
**问题**：`README.md` 只有 70 字节；Step 之间没有导航；没有一处列出「训练结果」（loss 曲线、PPL 数值、CEval 分数、GRPO 前后准确率对比）。
**影响**：招聘方或合作者 clone 下来看不出你「做成了什么」，只有「做了什么」。这是纯损失 —— 工作量已经足够，但成果不可见。
**建议**：
- 重写 README：一张阶段导航表（目录 → 学到的能力 → 对应产出/指标）
- 每个 Step 下加 `README.md`，贴关键结论：模型参数量、训练步数、PPL、CEval 分数、GRPO 格式/答案准确率曲线
- 把 `plot.ipynb` 里的曲线导出成 `assets/*.png` 并嵌入文档

### P1 — 可复现性
- 仓库**没有 `requirements.txt` / `environment.yml`**（Step4 有 `tokens.json` 63 B，疑似 HuggingFace token，请确认已失效或删除）
- 约 108 GB 数据与权重未上传，clone 后无法直接跑
- 建议：补 `requirements.txt`（或 `pyproject.toml`）；补 `data_asset_list.md`（引用 `.gitignore` 中的排除清单），并给每个数据集标注官方下载来源；权重给出 HF 链接或「如何从 0 复现」的步骤说明

### P2 — 能力补全
| 缺口 | 建议 |
|---|---|
| Reward Model 训练 | 现有 RL 全用规则/可验证奖励，补一个 reward model 训练（Bradley-Terry）会显著加分 |
| PPO 工程化 | 目前 PPO 只有 notebook 版，把 Step7 的 `rlhf-ppo/` 补成与 GRPO 同级的自研四模型 PPO |
| 推理侧工程 | vLLM 只当采样器用了；可补 continuous batching 原理、量化（GPTQ/AWQ）、投机解码的实测 |
| 多模态 / Agent | `train_tokenizer.py` 里已定义 `<tool_call>` / `<tool_response>` 特殊 token，但没有对应的工具调用数据与训练，说明有意图未落地 —— 这是很好的下一步 |
| 评测体系 | 目前有 PPL / CEval；可补 CMMLU、IFEval、MT-Bench，形成多维度对比表 |

### P3 — 仓库整洁度
- 目录名混用了空格（`Step0 pytorch`、`Step1 Transformer`、`Step6 Reinforcement Learnig`），其中 `Learnig` 是拼写错误（应为 `Learning`）；文件名也有中文与空格混排（`fenlei shoudong.py`、`minhash LSH.py`）。会给命令行操作与自动化脚本带来麻烦。
- 建议：如需重命名，用 `git mv` 保留历史；至少先把 `Learnig` 修掉。

---

## 六、如果你要用这个仓库求职，下一步最该做的三件事

1. **写 README + 每个 Step 的成果 README**（半天工作量，收益最大）
2. **给 Step7RL 单独写一份设计文档**：画一张「rollout 进程 ↔ 训练进程 ↔ 参考模型服务」的架构图，说明为什么这么设计（显存、吞吐、权重同步时机），并给出 GRPO vs DAPO vs GSPO 的实测对比曲线
3. **补 `requirements.txt` + 数据集来源清单 + 一份「从零复现指南」**

---

*本报告基于 GitHub 仓库 `mortal2341/learn-LLM` 在 2026-09-23 的文件树与源码内容生成。评分是依据代码证据密度的主观评估，用于定位能力分布，不等同于任何官方认证。*
