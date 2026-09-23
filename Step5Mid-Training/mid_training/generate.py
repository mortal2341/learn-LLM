import torch
import qwen3
from transformers import AutoTokenizer

# 加载分词器与模型 
model_path = "saved/midtrain/midtrain-qwen-0.1B/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = qwen3.Qwen3ForCausalLM.from_pretrained(model_path).to("cuda")


while True:
    prompt = input("User：")
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        repetition_penalty=1.2
    )
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("Bot：", response)
