# -*- coding: utf-8 -*-
# ---------------------------------
#  项目名称：预测脚本
# ---------------------------------

import json
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# model_name = "sft-plus-qwen-0.1B"
# local_dir  = "./saved/full_sft_plus/" + model_name
model_name = "dpo-qwen-0.1B"
local_dir  = "./saved/dpo/" + model_name
test_file = "data/dpo_test.jsonl"
out_file   = f"log/safe_dpo_{model_name}.json"

# 加载tokenizer和数据
tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
dataset = []
for line in open(test_file):
    dataset.append(json.loads(line))
print(f"Test Size: {len(dataset)}\n")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(local_dir)
model.eval()
device = "cuda"
model.to(device)

# 计算对数似然
def compute_log_likelihood(prompt: str, response: str) -> float:
    input_ids, labels_list = [], []
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    input_ids += [
        torch.tensor(prompt_ids + chosen_ids, dtype = torch.long)
    ]
    labels_list += [
        torch.tensor([-100]*len(prompt_ids) + chosen_ids, dtype = torch.long)
    ]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_tensor = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != tokenizer.pad_token_id)

    # 计算loss（负对数似然）
    with torch.no_grad():
        out = model(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels_tensor.to(device))
        mean_nll = out.loss.item()
    return np.exp(-mean_nll)

# 计算准确率
results = []
pbar = tqdm(dataset, total=len(dataset), desc="Test Safe Dataset for DPO")
pbar.set_postfix({'acc': '0.00%', 'correct': '0'})
correct = 0
total = 0

for sample in pbar:
    prompt = sample["prompt"]
    chosen = sample["pos_resp"]
    rejected = sample["neg_resp"]
    prompt = f"<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    # 计算正负答案的似然
    ll_chosen = compute_log_likelihood(prompt, chosen)
    ll_rejected = compute_log_likelihood(prompt, rejected)

    # 如果正例的似然大于负例，就算正确
    if ll_chosen > ll_rejected:
        correct += 1
    total += 1

    pbar.set_postfix({
        'acc': f'{(correct/total)*100:.2f}%',
        'correct': f'{correct}/{total}',
    })

    # 输出结果
    results.append({
        "prompt": prompt,
        "chosen_resp": chosen,
        "rejected_resp": rejected,
        "ll_chosen": ll_chosen,
        "ll_rejected": ll_rejected
    })

accuracy = correct / total
print(f"Preference Accuracy: {accuracy:.4f}")

pbar.close()
