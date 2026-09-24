# -*- coding: utf-8 -*-
# ---------------------------------------------------
# 项目名称: LoRA模型预测
# ---------------------------------------------------


import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = "cuda"
def predict(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 加载base模型的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B/", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B/", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_id="./output/lora_single_gpu/checkpoint-6670/").to(device)


PROMPT = """你是一个文本实体识别领域的专家，你需要从给定的句子中提取意图和实体. 请以指定的格式输出, 如 "音乐搜索-歌曲:千年等一回", "调高座椅温度-小数:0.27". 如果找不到任何意图和实体时, 输出"未知-无". """


# 预测部分
test_data = json.load(open("data/test_5k.jsonl"))
fw = open("data/test_5k_output.jsonl", "w")
test_result = []
for idx, line in tqdm(enumerate(test_data), total=len(test_data)):
    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{line['input']}"}
    ]

    response = predict(messages, model, tokenizer)
    line["predict"] = response
    test_result.append(line)
    line = json.dumps(line, ensure_ascii=False)
    fw.write(line + "\n")


# 计算指标的部分
intent_correct = 0
slot_correct = 0
intent_and_slot_correct = 0
total = 0
intent_result = {}
for line in test_result:
    gt = info["output"]
    pred = info["predict"]
    gt_intent, gt_slot = gt.split("-")
    pred_intent, pred_slot = pred.split("-")

    # 统计intent准确率
    if gt_intent == pred_intent:
        intent_correct += 1

    # 统计slots准确率
    gt_slot = gt_slot.split(",")
    pred_slot = pred_slot.split(",")
    if set(gt_slot) == set(pred_slot):
        slot_correct += 1

    # 统计intent+slots准确率
    if gt_intent == pred_intent and set(gt_slot) == set(pred_slot):
        intent_and_slot_correct += 1

    # 分意图统计
    if gt_intent not in intent_result:
        intent_result[gt_intent] = {"correct": 0, "total": 0}
    intent_result[gt_intent]["total"] += 1
    if gt_intent == pred_intent and set(gt_slot) == set(pred_slot):
        intent_result[gt_intent]["correct"] += 1

    total += 1


print("意图识别正确率: ", intent_correct/total)
print("槽位识别正确率: ", slot_correct/total)
print("意图+槽位整体识别正确率: ", intent_and_slot_correct/total)

print("="*50)
print("分意图的识别率：")
for intent, value in intent_result.items():
    print(f"{intent}: 准确率: {round(value['correct']/value['total'], 2)}")
