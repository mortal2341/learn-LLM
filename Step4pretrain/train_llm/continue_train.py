import datasets
import json
import math
import torch
import transformers
from transformers import AutoModelForCausalLM


context_length = 512
model_path = "saved/pretrain/pretrain-qwen-0.1B/"

# 加载预训练阶段的模型
model = AutoModelForCausalLM.from_pretrained(model_path)
model_size = sum(t.numel() for t in model.parameters())
print(f"Model Size: {model_size/1000**2:.1f}M parameters")

# 加载预训练数据
raw_datasets = datasets.load_dataset(
    "json", data_files="data/processed_medical_mix.jsonl"
)
raw_datasets = raw_datasets["train"].train_test_split(test_size=0.01, seed=42)
print("dataset info")
print(raw_datasets)

# 保存测试集用于后续评估
raw_datasets["test"].to_json(path_or_buf = './data/continue_train_test_6k.json', force_ascii=False)


# 加载自定义Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "tokenizers/merged_tokenizer"
)

# 预处理数据
def tokenize(element):
    outputs = tokenizer(
        element["text"],
        add_special_tokens=False
    )
    input_ids_list = outputs['input_ids']
    new_input_ids_list, new_attn_mask_list = [], []
    for input_ids in input_ids_list:
        input_ids_eos = input_ids[:context_length-1] + [tokenizer.eos_token_id]
        new_input_ids_list.append(input_ids_eos)
        new_attn_mask_list.append([1] * len(input_ids_eos))
    return {
        "input_ids": new_input_ids_list,
        "attention_mask": new_attn_mask_list
    }

tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, num_proc=16, remove_columns=raw_datasets["train"].column_names
)
print("tokenize dataset info")
print(tokenized_datasets)
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)


# 训练参数 
args = transformers.TrainingArguments(
    output_dir="saved/continue_pretrain",
    per_device_train_batch_size=16,  # 每个GPU的训练batch数
    per_device_eval_batch_size=16,  # 每个GPU的测试batch数
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=10,
    gradient_accumulation_steps=8,  # 梯度累计总数
    num_train_epochs=1,  # 训练epoch数
    weight_decay=0.1,    # 权重衰减比例
    warmup_ratio=0.03,    # warmup 比例
    optim="adamw_torch",  # 优化器使用adamw
    lr_scheduler_type="cosine",  # 学习率衰减策略
    learning_rate=2e-5,  # 基础学习率，
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
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# 保存模型和tokenizer
model.save_pretrained("./saved/continue_pretrain/cpt-qwen-0.1B/")  # 保存模型的路径
tokenizer.save_pretrained("./saved/continue_pretrain/cpt-qwen-0.1B/")


# 保存训练日志
with open("log/continue_running_states.json", "w", encoding="utf-8") as fw:
    print(trainer.state.log_history, file=fw)

# 根据提示词生成答案
pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
print("GENERATE:", pipe("人工智能", num_return_sequences=1)[0]["generated_text"])
print("GENERATE:", pipe("糖尿病", num_return_sequences=1)[0]["generated_text"])
