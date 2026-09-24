# -*- coding: utf-8 -*-
# ---------------------------------------------------
# 项目名称: LoRA Adapter模型合并 
# ---------------------------------------------------


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 合并后的路径
output_path = "Qwen/Qwen3-8B-SFT/"

# 加载base模型的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B/", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B/", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_id="./output/lora_single_gpu/checkpoint-6670/")

# 合并权重
print("开始合并LoRA权重...")
merged_model = model.merge_and_unload()

# 保存合并后的模型
print(f"保存合并模型到: {output_path}")
merged_model.save_pretrained(output_path)

# 保存分词器
tokenizer.save_pretrained(output_path)
