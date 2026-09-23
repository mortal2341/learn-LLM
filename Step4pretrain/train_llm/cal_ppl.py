# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: 利用滑动窗口计算文档PPL 
# --------------------------------------------


import json
import torch
import datasets
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizers/merged_tokenizer")
model = AutoModelForCausalLM.from_pretrained(
    "saved/continue_pretrain/cpt-qwen-0.1B/",
).to("cuda")

# 加载预训练数据
test_data = datasets.load_dataset(
    "json", data_files="data/pretrain_test_1w.json"
)

stride = 256
max_length = 512
total_ppls = []
for line in tqdm(test_data["train"], total=len(test_data)):
    text = line["text"]
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):

        # 计算要预测的区间起点和终点
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        # 得到这一段的loss
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()  # 统计target里面有多少token 
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # 减去因为内部label right-shift的token数

        # 累计loss 和 tokens
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # 对所有的交叉墒损失做平均 
    ppl = torch.exp(avg_nll)
    total_ppls.append(ppl.item())

average_ppl = sum(total_ppls) / len(total_ppls)
print(f"文档数: {len(total_ppls)}, 平均PPL: {average_ppl}")
