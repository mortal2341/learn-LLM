# qwen3_intent_slot_filling：意图 + 槽位联合抽取

基于 **Qwen3-8B** 的 LoRA 微调实战，完成车载语音场景下「意图识别 + 槽位填充」联合抽取任务。

## 任务说明

给定一条用户自然语言输入，模型输出 `意图-槽位` 结构化结果：

| 输入 | 输出 |
| --- | --- |
| 前座通风能不能开到最大？ | `调大座椅通风-位置:前排,强度:最强` |
| 帮我放一首千年等一回 | `音乐搜索-歌曲:千年等一回` |
| 把空调风量调到最小 | `设置空调风力-强度:最低` |
| 你型我想的是 | `未知-无` |

- 意图示例：调大座椅通风、音乐搜索、设置空调风力、关闭导航、调高座椅温度、新闻搜索 …
- 槽位示例：位置（前排 / 主驾 / 主对角）、强度（最强 / 最低 / 最高）、歌曲名、POI、新闻人物、小数 …

## 目录结构

```
qwen3_intent_slot_filling/
├── Qwen3-LoRA-Finetune.ipynb     # LoRA 微调 Notebook（单卡）
├── train_ner.py                  # 微调训练脚本（DeepSpeed 多卡）
├── train_ds.sh                   # DeepSpeed 4 卡训练启动脚本
├── predict.py                    # LoRA 模型预测 + 指标计算
├── predict_vllm.py               # vLLM 服务并行推理 + 指标计算
├── merge_model.py                # LoRA Adapter 合并回基座模型
├── download.sh                   # 从 ModelScope 下载 Qwen3-8B
├── vllm_server.sh                # 启动 vLLM 推理服务
├── data/
│   ├── raw_data.jsonl            # 原始训练语料（未纳入 Git）
│   ├── test_5k.jsonl             # 测试集（小型样例）
│   ├── test_5k_output.jsonl      # predict.py 推理输出
│   └── test_5k_vllm_output.jsonl # vLLM 推理结果
├── output/                       # 训练输出（LoRA 检查点配置；权重未纳入 Git）
│   ├── lora_multi_gpu/checkpoint-1668/
│   └── lora_single_gpu/checkpoint-6670/
└── Qwen/                         # 基座 / 合并模型目录（本地生成，未纳入 Git）
    ├── Qwen3-8B/
    └── Qwen3-8B-SFT/
```

## 训练流程

1. 下载基座模型：`bash download.sh`（ModelScope 下载 Qwen3-8B）
2. 准备数据：`data/raw_data.jsonl`，每行格式 `{"input": "...", "output": "意图-槽位"}`
3. 微调训练：`bash train_ds.sh`（4 卡 DeepSpeed），或运行 `Qwen3-LoRA-Finetune.ipynb`
4. 合并权重：`python merge_model.py` → `Qwen/Qwen3-8B-SFT/`
5. 推理评估：`bash vllm_server.sh` 启动服务后，运行 `python predict_vllm.py` 计算指标

## LoRA 关键配置

- 基座模型：`Qwen3-8B`，`dtype=bfloat16`
- `r=8`、`lora_alpha=32`、`lora_dropout=0.1`
- `target_modules`：`q/k/v/o_proj` + `gate/up/down_proj`（Attention + FFN）
- 训练参数：`lr=1e-4`、`epochs=2`、cosine 学习率调度、`max_length=256`

## 说明

模型权重（`*.safetensors` / `*.pt` / `*.pth`）、分词器产物（`vocab.json` / `merges.txt`）及大体积训练语料（`raw_data.jsonl`）均未纳入 Git，需按上述流程在本地生成 / 下载。
