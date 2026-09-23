# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: DPO预测代码
# --------------------------------------------


import json
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import PROMPT

model_name_or_path = 'output/phi-4-sft/merged_phi4_sft/'
lora_path = "output/dpo-phi4-adgen/checkpoint-312/"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)

model = PeftModel.from_pretrained(
    model,
    lora_path,
    torch_dtype=torch.bfloat16
)

# 合并模型
# model = model.merge_and_unload()
# model.save_pretrained("./output/phi-4-dpo/merged_phi4_dpo")
# tokenizer.save_pretrained("./output/phi-4-dpo/merged_phi4_dpo")

device = "cuda"
model.to(device)


while True:
    inputs = input("输入: ")

    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": inputs}
    ]

    # 调用模型进行对话生成
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to(model.device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("输出: ", response)
