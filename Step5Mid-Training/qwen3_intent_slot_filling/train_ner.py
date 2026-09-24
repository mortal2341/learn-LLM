# -*- coding: utf-8 -*-
# ---------------------------------------------------
# 项目名称: 案例实战：大模型意图+槽位联合抽取
# ---------------------------------------------------


import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq


# 指令微调的提示词
PROMPT = """你是一个文本实体识别领域的专家，你需要从给定的句子中提取意图和实体. 请以指定的格式输出, 如 "音乐搜索-歌曲:千年等一回", "调高座椅温度-小数:0.27". 如果找不到任何意图和实体时, 输出"未知-无". """


def process_func(example):
    """
    将数据集进行预处理
    """

    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=128
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]     
    return response


# 加载模型权重
model_dir = "Qwen/Qwen3-8B" 
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)

# 加载、处理数据集和测试集
dataset_path = "data/raw_data.jsonl"

# 得到训练集，验证集和测试集
test_val_ratio = 0.05
total_df = pd.read_json(dataset_path)

train_df = total_df[int(len(total_df) * test_val_ratio):]
test_val_df = total_df[:int(len(total_df) * test_val_ratio)]
val_df = test_val_df[:int(len(test_val_df))//5]
test_df = test_val_df[int(len(test_val_df))//5:]
test_df.to_json('data/test_5k.jsonl', orient='records', force_ascii=False)


print("train size:", len(train_df))
print("eval size:", len(val_df))
print("test size:", len(test_df))

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

train_dataset = train_ds.map(process_func, num_proc=16, remove_columns=train_ds.column_names)
val_dataset = val_ds.map(process_func, num_proc=16, remove_columns=val_ds.column_names)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Attention + FFN
    inference_mode=False,  # 训练模式
    r=8,  # Lora rank
    lora_alpha=32,  # Lora Alpha
    lora_dropout=0.1,  # Dropout比例
)

args = TrainingArguments(
    output_dir="./output/lora_multi_gpu",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    weight_decay=0.1,    # 权重衰减比例
    max_grad_norm=1.0,   # 梯度裁剪最大值
    warmup_ratio=0.01,    # warmup 比例
    optim="adamw_torch",  # 优化器使用adamw
    lr_scheduler_type="cosine",  # 学习率衰减策略
    num_train_epochs=2,
    save_steps=500,
    save_total_limit=3,
    learning_rate=1e-4,
    label_names=["labels"]
)

model = get_peft_model(model, config)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()


# 保存训练日志
with open("output/running_states.json", "w") as fw:
    print(trainer.state.log_history, file=fw)


# 用测试集的随机20条，测试模型
sample_test = test_df.sample(n=20)
test_text_list = []
for index, row in sample_test.iterrows():
    input_value = row['input']
    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{input_value}"}
    ]
    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": f"{response}"})
    print(messages[1]['content'], "=>", messages[2]['content'])
