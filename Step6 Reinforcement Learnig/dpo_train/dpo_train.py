# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: DPO偏好对齐训练代码
# --------------------------------------------


import re
import json
import math
import torch
import datasets
import transformers
import os
import copy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq
from trl import DPOConfig, DPOTrainer


model_path = "saved/full_sft_plus/sft-plus-qwen-0.1B/"

# 加载SFT阶段的模型
model = AutoModelForCausalLM.from_pretrained(model_path)
model_size = sum(t.numel() for t in model.parameters())
print(f"Model Size: {model_size/1000**2:.1f}M parameters")

# 加载自定义Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

# 预处理数据
def process_dataset(dataset_path):
    examples = []
    output_text = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }
    raw_data = open(dataset_path, 'r').readlines()
    for line in tqdm(raw_data, total=len(raw_data), desc="Processint Data"):
        example = json.loads(line)
        # apply chat template
        prompt = f"<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        # prompt + chosen, prompt + rejected
        chosen = f"{example['pos_resp']}<|im_end|>"
        rejected = f"{example['neg_resp']}<|im_end|>"
        output_text["prompt"].append(prompt)
        output_text["chosen"].append(chosen)
        output_text["rejected"].append(rejected)

    dataset = datasets.Dataset.from_dict(output_text)
    dataset = dataset.shuffle()
    return dataset

train_dataset = process_dataset("./data/dpo_train.jsonl")
print("tokenize dataset info")
print(train_dataset)



# 训练参数配置
training_args = DPOConfig(
    output_dir="saved/dpo",
    per_device_train_batch_size=6,  # 每个GPU的训练batch数
    gradient_accumulation_steps=8,  # 梯度累计总数
    overwrite_output_dir=True,
    logging_steps=10,
    learning_rate=1e-6,  # 这个一般不要设置太大，容易过拟合或者train飞
    weight_decay=0.1,    # 权重衰减比例
    max_grad_norm=1.0,   # 梯度裁剪最大值
    warmup_ratio=0.01,    # warmup 比例
    optim="adamw_torch",  # 优化器使用adamw
    lr_scheduler_type="cosine",  # 学习率衰减策略
    num_train_epochs=1,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
)


# 训练参数 
print("Train Args:")
print(training_args)


# 初始化Trainer
trainer = DPOTrainer(
    model=model,                     # 策略模型
    ref_model=copy.deepcopy(model),  # 参考模型
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

trainer.train()

# 保存模型和tokenizer
model.save_pretrained("./saved/dpo/dpo-qwen-0.1B/")  # 保存模型的路径
tokenizer.save_pretrained("./saved/dpo/dpo-qwen-0.1B/")


# 保存训练日志
with open("log/full_dpo_running_states.json", "w") as fw:
    print(trainer.state.log_history, file=fw)
