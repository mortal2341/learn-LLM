# Step6 · 强化学习地基与对齐实战

**这个阶段解决问题**：先从零手写经典 RL 算法（能推导 PPO，而不是只会调 TRL），再对自训的 0.1B 模型做 DPO 对齐，最后落地一个真实业务项目（电商广告文案）。

## 模型链位置

本步完成模型链最后一环：`sft-plus-qwen-0.1B` → **DPO** → `dpo-qwen-0.1B`。

## 子模块

### `rl-base/` —— 从零手写经典 RL

| 文件 | 算法 |
|---|---|
| `REINFORCE.ipynb` | 策略梯度蒙特卡洛 |
| `ActorCritic.ipynb` | Actor-Critic 架构 |
| `DQN.ipynb` | 深度 Q 网络（含经验回放 / 目标网络） |
| `MonteCarlo.ipynb` / `TrainMonteCarlo.ipynb` | 蒙特卡洛价值估计与训练 |
| `TD.ipynb` | 时序差分学习 |
| `PPO.ipynb` | PPO：clip 目标 + GAE |
| `DPO_loss.py` | 手写 DPO 损失（不调 TRL，对照推导） |
| `utils.py` | 公共工具（环境封装等） |

学习路径：REINFORCE → AC → TD → DQN → PPO，每个算法都是「公式 → 代码 → 玩具环境验证」。

### `dpo/` —— 偏好优化三件套
- `dpo.ipynb` / `orpo.ipynb` / `apo.ipynb`：DPO / ORPO / APO 三种偏好优化算法的原理推导与实现对比
- `test.ipynb`：验证实验

### `dpo_train/` —— 自家模型 DPO 全流程

- 基座：**`sft-plus-qwen-0.1B`（Step5 自训产物）**，`ref_model = copy.deepcopy(model)`
- 框架：TRL `DPOTrainer`，lr 1e-6 / 1 epoch / bs 6×8 / cosine
- 数据：`data/dpo_train.jsonl` 偏好对（prompt + pos_resp / neg_resp，chat template 手工拼接）
- 评测：`dpo_evaluate.py`

**训练成果**（真实数据，`log/full_dpo_running_states.json`）：

| 指标 | 起点 | 终点 |
|---|---|---|
| DPO loss | 0.496（step 10） | **0.0003**（step 800） |
| rewards / margins | 0.55 | **15.33** |
| 训练时长 | — | 809 步 / 18.8 min |

margin 从 0.55 拉到 15.3，策略模型与参考模型对偏好对的区分度持续放大，对齐生效且未崩溃（loss 单调下降无回升）。

### `ad_text_gen/` —— 电商广告文案生成（业务落地项目）

完整业务闭环，基于 Phi-4 + LlamaFactory / ms-swift：

1. `prompts.py`：结构化电商文案 prompt（角色 / 任务 / 关键词格式 / 输出要求）
2. `Phi4-Lora_Finetune.ipynb`：Phi-4 LoRA SFT
3. `generate_chosen_data.ipynb` / `generate_rejected_data.ipynb`：**用 vLLM 批量自造 chosen / rejected 偏好对**
4. `Phi4-DPO-Train.ipynb`：DPO 训练（产出 `checkpoint-312`）
5. `cli.py`：加载 SFT 基座 + DPO LoRA 的交互式推理入口

<!-- TODO: 在此补充 2-3 组 DPO 前后的广告文案生成对比样例，这是面试最好讲的素材 -->
