# Step4 · 预训练全流程（词表 → 数据 → 训练 → 评测）

**这个阶段解决的问题**：从零训练一个 0.1B 中文模型。不是「跑通 Trainer」，而是词表、数据清洗、质量筛选、去重、训练、评测全部自己做。

## 产出模型

`pretrain-qwen-0.1B` —— Qwen3 架构，16 层 / hidden 512 / 8 Q heads / 4 KV heads / vocab 58,703 / ctx 512，230 万条中文语料从零训练。

## 训练成果（真实日志数据，见 `train_llm/log/`）

| 指标 | 起点 | 终点 |
|---|---|---|
| 训练 loss | 11.01（step 10） | 2.63（step 7,250+） |
| 验证 loss | 5.68（step 500） | **2.66（step 7,000）** |
| 困惑度 PPL | — | **≈ 14.2**（exp(2.66)） |

eval_loss 全程单调下降无回升，曲线数据：`train_llm/log/pretrain_running_states.json`，可视化：`train_llm/plot.ipynb`。

<!-- TODO: 在此填入 lm-evaluation-harness 跑出的 CEval 各科目分数 -->

## 子模块

### `custom_tokenizer/` —— 自训 BBPE 词表
- `train_tokenizer.py`：从 160 万条语料训练 **BBPE 词表（vocab 9,600）**，ByteLevel pre-tokenizer，完整设计特殊 token（`<|im_start|>` / `<|im_end|>` / `<think>` / `<answer>` / `<tool_call>` 等）
- `merge_tokenizer.py`：自训词表与 Qwen 词表合并 → 最终 58,703
- `stats.py`：词表字符分布统计
- `config.json`：tokenizer 配置

### `prepare_data/` —— 多源语料工程
- `download.sh`：统一管理全部数据集下载（wiki 百科 / firefly / belle / R1 蒸馏数据等，走 hf-mirror）
- `format_raw_data.py` / `format_raw_data_2048.py`：512 与 2048 两种上下文长度的格式化
- `format_raw_data_sft_2048.py` / `format_raw_data_think_distill_2048.py`：SFT 与思维链数据的分流预处理

### `quality_model/` —— 数据质量筛选
- `main.ipynb`：基于 `bert-base-chinese` 训练数据质量分类器
- `gen_data.ipynb`：质量分类训练数据构造
- `deduplicate.ipynb`：语料去重

### `minhash LSH.py` —— 手写 MinHash LSH 去重
不调 datasketch 库，手写 MinHash 签名 + LSH 分桶的近似去重。

### `data_js/` —— Data-Juicer 工业流水线
- `data_process_js.yaml` / `data_process_js_2048.yaml`：Data-Juicer 数据处理配置
- `log_2048.txt`：真实运行日志（288 KB）

### `train_llm/` —— 训练与评测
- `pre_train.py`：0.1B 从零训练（自定义 Qwen3Config；lr 3e-4 cosine / warmup 10% / bf16 / 2 epoch / bs 16×8 / wd 0.1）
- `continue_train.py`：继续预训练
- `cal_ppl.py`：困惑度计算
- `generate.py`：生成测试
- `lm-evaluation-harness-main/` + `test_ceval.sh`：接 lm-eval-harness 跑 CEval 中文评测
- `plot.ipynb`：训练曲线可视化

## 数据流

```
多源下载(download.sh)
  → 格式化(format_raw_data*.py, 512/2048)
  → 质量筛选(quality_model: BERT 分类器)
  → 去重(minhash LSH.py / deduplicate.ipynb)
  → Data-Juicer 精处理(data_js/)
  → 预训练(pre_train.py, 230w 条)
  → 评测(cal_ppl.py / test_ceval.sh)
```
