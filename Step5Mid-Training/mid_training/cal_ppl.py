# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: 利用滑动窗口计算文档PPL 
# --------------------------------------------


import json
import torch
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer
import qwen3


# model_path = "saved/pretrain/pretrain-qwen-0.1B/"
model_path = "saved/midtrain/midtrain-qwen-0.1B/"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = qwen3.Qwen3ForCausalLM.from_pretrained(
    model_path,
).to("cuda")

# 加载测试数据
test_data = datasets.load_dataset(
    "json", data_files="data/multifieldqa_zh.jsonl"
)

stride = 256#512 
max_length = 8192#2048,4096 
total_ppls = []
for text in tqdm(test_data["train"]["context"], total=len(test_data)):
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
