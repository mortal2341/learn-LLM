# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: 自定义Qwen3大模型继续预训练 
# --------------------------------------------

import datasets
import json
import math
import torch
import transformers
import qwen3


# 重新定义模型的结构参数
context_length = 2048
pretrain_path = "saved/pretrain/pretrain-qwen-0.1B/"
# config = transformers.AutoConfig.from_pretrained(
config = qwen3.Qwen3Config.from_pretrained(
    pretrain_path,
    bos_token_id=1,
    eos_token_id=2,
    hidden_size=512,
    intermediate_size=1536,
    max_position_embeddings=8192,
    num_attention_heads=8,
    num_key_value_heads=4,
    num_hidden_layers=16,
    n_ctx=context_length, #上下文长度从512改成2048
    rope_theta=100000, #调大基频, ABF
    vocab_size=58703,
)

# model = transformers.models.Qwen3ForCausalLM(config)
model = qwen3.Qwen3ForCausalLM(config)
print("Model Summary:")
print(model)
print("=" * 100)

# 打印模型参数
model_size = sum(t.numel() for t in model.parameters())
print("Model Config:")
print(config)
print(f"Model Size: {model_size/1000**2:.1f}M parameters")
print("=" * 100)

# 加载预训练模型
# pretrain_model = transformers.AutoModelForCausalLM.from_pretrained(
pretrain_model = qwen3.Qwen3ForCausalLM.from_pretrained(
    pretrain_path,
    torch_dtype=torch.bfloat16)
model.load_state_dict(pretrain_model.state_dict())


# 加载预训练数据
raw_datasets = datasets.load_dataset(
    "json", data_files="data/processed_mid_training_80w.jsonl"
)

raw_datasets = raw_datasets["train"].train_test_split(test_size=0.005, seed=42)
print("dataset info")
print(raw_datasets)

# 保存测试数据
raw_datasets["test"].to_json(path_or_buf = './data/midtrain_test_4k.json', force_ascii=False)


# 加载分词器
tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrain_path,
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
print("=" * 100)

# 把一个batch做padding
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)


# 训练参数
args = transformers.TrainingArguments(
    output_dir="saved/midtrain",
    per_device_train_batch_size=4, # 每个gpu的训练batch数  
    per_device_eval_batch_size=4,  # 每个gpu的测试batch数
    eval_strategy="steps", 
    eval_steps=500,
    logging_steps=10,
    gradient_accumulation_steps=8, # 梯度的累积步数
    num_train_epochs=1, # 训练的epoch数
    weight_decay=0.1,   # weight decay的比率
    warmup_ratio=0.02,   # warmup的比例  要比 pretrain 小很多 pretrain大概是10%
    optim="adamw_torch",  # 优化器选择的adamW
    lr_scheduler_type="cosine", # 学习率的衰减策略，[0, T/4]
    learning_rate=5e-5, # 学习率, [1e-4, 5e-5], 
    save_steps=500,
    save_total_limit=2, # 最大保存2个ckpt
    bf16=True,  # 开启bf16训练，对于Amper架构以下的显卡建议替换为fp16
)
print("Train Args:")
print(args)
print("=" * 100)

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

# 做一次模型评估
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# 保存模型和tokenizer
model.save_pretrained("./saved/midtrain/midtrain-qwen-0.1B/")  # 保存模型的路径
tokenizer.save_pretrained("./saved/midtrain/midtrain-qwen-0.1B/")


# 保存训练日志
with open("log/midtrain_running_states.json", "w", encoding="utf-8") as fw:
    print(trainer.state.log_history, file=fw)


# 根据提示词生成答案
pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
print("GENERATE:", pipe("人工智能", num_return_sequences=1)[0]["generated_text"])
