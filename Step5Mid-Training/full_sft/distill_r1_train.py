# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: SFT-Distill-Reasoning全参训练代码
# --------------------------------------------


import re
import json
import math
import torch
import datasets
import transformers
import os

os.environ['HF_HOME'] = "/root/autodl-tmp/huggingface" 
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq


context_length = 2048
model_path = "saved/full_sft_plus/sft-plus-qwen-0.1B/"

# 加载预训练阶段的模型
model = AutoModelForCausalLM.from_pretrained(model_path)
model_size = sum(t.numel() for t in model.parameters())
print(f"Model Size: {model_size/1000**2:.1f}M parameters")

# 加载预训练数据
raw_datasets = datasets.load_dataset(
    "json", data_files="data/processed_distill_r1_7w.jsonl"
)
raw_datasets = raw_datasets["train"].train_test_split(test_size=0.01, seed=42)
print("dataset info")
print(raw_datasets)

# 保存测试集用于后续评估
raw_datasets["test"].to_json(path_or_buf = './data/distill_r1_test_700.json', force_ascii=False)


# 加载自定义Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

# 预处理数据
def process_func(elements):

    system_message = "You are a helpful assistant"
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
    IGNORE_TOKEN_ID = -100
    
    im_start = tokenizer.bos_token_id
    im_end = tokenizer.eos_token_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    messages = elements["conversation"]

    # 拼接多轮对话模版
    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [IGNORE_TOKEN_ID] * len(system)
    assert len(input_id) == len(target)
    for j, sentence in enumerate(messages):
        role = roles[sentence["role"]]
        _input_id = tokenizer(role).input_ids + nl_tokens + \
            tokenizer(sentence["content"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == '<|im_start|>user':
            _target = [IGNORE_TOKEN_ID] * len(_input_id)
        elif role == '<|im_start|>assistant':
            _target = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
        else:
            raise Exception("undefine role!")
        target += _target
    assert len(input_id) == len(target)

    # 按最大截断
    input_id = input_id[:context_length]
    target = target[:context_length]
    mask = [int(k != tokenizer.pad_token_id) for k in input_id]

    return dict(
        input_ids=input_id,
        labels=target,
        attention_mask=mask,
    )

tokenized_datasets = raw_datasets.map(
    process_func, num_proc=16,
    remove_columns=raw_datasets["train"].column_names
)
print("tokenize dataset info")
print(tokenized_datasets)

# 做batch的padding
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)


# 训练参数 
args = transformers.TrainingArguments(
    output_dir="saved/think",
    per_device_train_batch_size=4,  # 每个GPU的训练batch数
    per_device_eval_batch_size=8,  # 每个GPU的测试batch数
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=10,
    gradient_accumulation_steps=8,  # 梯度累计总数
    num_train_epochs=5,  # 训练epoch数
    weight_decay=0.1,    # 权重衰减比例
    max_grad_norm=1.0,   # 梯度裁剪最大值
    warmup_ratio=0.01,    # warmup 比例
    optim="adamw_torch",  # 优化器使用adamw
    lr_scheduler_type="cosine",  # 学习率衰减策略
    learning_rate=5e-5,  # 基础学习率，
    save_steps=500,
    save_total_limit=2,
    bf16=True,  # 开启bf16训练, 对于Amper架构以下的显卡建议替换为fp16=True
)
print("Train Args:")
print(args)

# 开始训练 
trainer = transformers.Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
trainer.train()

# 保存模型和tokenizer
model.save_pretrained("./saved/think/think-qwen-0.1B/")  # 保存模型的路径
tokenizer.save_pretrained("./saved/think/think-qwen-0.1B/")


# 保存训练日志
with open("log/think_running_states.json", "w", encoding="utf-8") as fw:
    print(trainer.state.log_history, file=fw)
