# Step7 · 后训练阶段：自研 RL 训练框架

**这个阶段解决的问题**：不依赖 verl / TRL 等现成 RL 框架，自己搭建一套 **vLLM rollout 与 DeepSpeed 训练解耦**的分布式 GRPO / DAPO / GSPO 训练系统，并扩展出 on-policy 蒸馏、医疗 RL 等方向。

## `rlhf_grpo/` —— 自研 GRPO / DAPO / GSPO 框架（核心）

### 系统架构

```
┌─────────────────┐  mp.Queue(权重热更新)  ┌──────────────────┐
│  rollout 进程     │ ◄────────────────── │  训练进程           │
│  vLLM (GPU 1)    │                     │  DeepSpeed 引擎     │
│  · n=8 采样/prompt│                     │  · GRPO/DAPO/GSPO  │
│  · 计算采样 logp  │      HTTP(二进制)     │  · ZeRO + bf16     │
└────────┬────────┘ ──────────────────► └──────────────────┘
         │            ref_server.py
         ▼         (bottle+tornado)
  reward 计算        参考模型 logp
  (正确性+格式)      (独立 GPU 常驻)
```

- **rollout 进程**（`gen_worker`）：vLLM 负责 8 条/prompt 的组采样 + 采样模型 logp 计算；清理 NCCL 环境变量避免与训练进程冲突；采样模型每 N 步从队列热更新权重（`load_weights`）
- **训练进程**：DeepSpeed 引擎做三种算法更新；`get_batch` 从参考模型服务拉取组装好的训练 batch
- **参考模型服务**（`ref_server.py`）：bottle + tornado 自建 HTTP 服务，LIFO 队列，二进制 bytes 协议传 tensor（比 JSON/base64 快），支持 tensor/string 两种模式自动轮换

### 三种算法实现要点（`train.py`）

| 算法 | 关键实现 |
|---|---|
| GRPO | 组内优势归一化 `(r-mean)/(std+ε)`；token 级 clip；K3 无偏 KL 估计 `exp(Δ)-Δ-1`；β 加权约束 |
| DAPO | **Clip-Higher**（上下 clip 不对称 0.2/0.28）；**去掉 KL 约束**；**Token-Level Policy Gradient Loss**（全局 token 平均而非样本平均）；软长度惩罚（L_max/L_cache） |
| GSPO | **序列级重要性采样**：`s_i = exp(mean(log p - log p_old))`，序列级 clip，替代 token 级 ratio |

通用工程细节：
- **无区分度组自动丢弃**：组内 reward 极差 < 1e-4（全对/全错）的样本直接跳过，不浪费训练
- 循环逐行计算 log_softmax，**降低峰值显存**
- rollout 全量落盘 `saved/reasoning_log.txt`（107 KB 已保留），用于 badcase 分析
- 每 20 步在 100 条测试集上评估 **格式准确率 + 答案准确率** 双指标

### 训练配置（`config/config.yaml`）

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-7B（bf16，sdpa 注意力） |
| 数据 | openai/gsm8k |
| 采样 | temperature 0.9 / max 700 tokens / 每组 8 条 |
| 训练 | 500 步 / clip 0.2 / β 0.04 / 每 16 步更新采样模型 |

<!-- TODO: 填入 GRPO 训练前后的 format accuracy / answer accuracy 对比（训练日志的 [Evaluation] 行）——这是本框架最有说服力的成果数字 -->

### 启动方式

```bash
# 1. 先起参考模型服务
python ref_server.py
# 2. 再起训练（含 rollout 子进程）
deepspeed train.py
```

## 其他方向

| 目录 | 内容 |
|---|---|
| `on_policy_distill/` | on-policy 蒸馏：教师模型服务（`teacher_server.py`）+ 学生训练（18 KB `train.py`） |
| `rlhf-ppo/` | RLHF-PPO 完整 notebook 实现（168 KB，四模型架构：policy / ref / reward / value） |
| `grpo_medical/` | GRPO 医疗领域应用：Unsloth LoRA 版本，含**单卡 32G 可跑**的优化版（`grpo_medical_lora_128_32G.ipynb`） |
| `r_zero/` | 基于 verl 框架 + 自研题目生成（`question_generate`）与判分（`question_evaluate`）的 RL 数据闭环 |
| `train_gold/` | 蒸馏训练入口（`train_distill.py`） |
