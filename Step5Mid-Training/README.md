# Step 5：Mid-Training（中期训练）

记录大模型**中期训练 / 后训练**阶段的学习与实战，涵盖数据准备、全量 SFT、LoRA/QLoRA 参数高效微调、继续预训练（Mid-Training）、推理蒸馏、模型合并，以及一个完整的「意图识别 + 槽位填充」实战项目。

## 目录结构

| 目录 | 说明 |
| --- | --- |
| `prepare_data/` | 数据准备与格式化（raw → 2048 / 512 / SFT / think-distill 多种格式） |
| `data_js/` | 数据清洗 / 处理流水线（YAML 配置 + 运行脚本） |
| `full_sft/` | 全量参数 SFT 微调 |
| `lora_qlora/` | LoRA / QLoRA（NF4）参数高效微调 |
| `mid_training/` | Mid-Training / 继续预训练（含 PPL 评估） |
| `distill_reasoning/` | 推理能力蒸馏 |
| `model_merging/` | 模型合并 |
| `qwen32b_cot/` | Qwen3-2B 思维链（CoT）实战 |
| `qwen3_intent_slot_filling/` | 实战：Qwen3-8B 意图识别 + 槽位填充（LoRA） |

## 子项目简介

### qwen3_intent_slot_filling（意图 + 槽位联合抽取）

基于 **Qwen3-8B** 做 LoRA 微调，完成车载语音场景下「意图识别 + 槽位填充」联合抽取任务：

- 输入：一句自然语言 query（如「前座通风能不能开到最大？」）
- 输出：`意图-槽位` 结构化结果（如 `调大座椅通风-位置:前排,强度:最强`）
- 技术点：LoRA 微调、DeepSpeed 多卡训练、vLLM 推理加速、LoRA 权重合并

详细说明见 [qwen3_intent_slot_filling/README.md](qwen3_intent_slot_filling/README.md)。

## 说明

- 本目录仅收录**代码 / 配置 / Notebook / 小型样例**。
- 模型权重（`*.safetensors` / `*.pt` / `*.pth`）、分词器产物（`vocab.json` / `merges.txt` / `tokenizer.json`）、训练日志与大体积语料均不纳入 Git 版本控制。
- 排除规则见仓库根目录 `.gitignore` 与被排除资产清单 `excluded_large_files.txt`。
