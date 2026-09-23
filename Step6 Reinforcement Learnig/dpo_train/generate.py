# -*- coding: utf-8 -*-
# ---------------------------------
#  项目名称：模型预测代码(流式输出)
# ---------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


# 加载分词器与模型 
model_path = "saved/dpo/dpo-qwen-0.1B/"
# model_path = "saved/full_sft_plus/sft-plus-qwen-0.1B/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


while True:
    prompt = input("User：")
    conversation = []
    conversation.append({"role": "user", "content": prompt})
    templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
    prompt_template = tokenizer.apply_chat_template(**templates)
    inputs = tokenizer(prompt_template, truncation=True, return_tensors="pt").to(model.device)

    print("Bot: ", end="")
    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=2048,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        streamer=streamer,
        repetition_penalty=1.1
    )
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):])
    print("\n")
