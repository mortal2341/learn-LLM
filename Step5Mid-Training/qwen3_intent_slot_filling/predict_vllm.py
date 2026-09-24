# -*- coding: utf-8 -*-
# ---------------------------------------------------
# 项目名称: 并行调用vLLM模型服务+指标计算 
# ---------------------------------------------------


import os
import json
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed 


MAX_WORKERS = 64
PROMPT = """你是一个文本实体识别领域的专家，你需要从给定的句子中提取意图和实体. 请以指定的格式输出, 如 "音乐搜索-歌曲:千年等一回", "调高座椅温度-小数:0.27". 如果找不到任何意图和实体时, 输出"未知-无". """

llm_client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

def request_server(line):
    try:
        input_text = line["input"] 
        completion = llm_client.chat.completions.create(
            model="qwen3-intent",
            messages = [
                {"role": "system", "content": f"{PROMPT}"},
                {"role": "user", "content": f"{input_text}"}
            ],
            # qwen3 vllm调用，enable_thinking的开关要如下这样设置
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response = completion.choices[0].message.content
        line["predict"] = response
        return line 
        
    except:
        return ""



test_data = json.load(open("data/test_5k.jsonl"))
fw = open("data/test_5k_vllm_output.jsonl", "w")

# 进程池调用
test_result = []
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(request_server, line) for line in test_data]
    for future in tqdm(as_completed(futures), total=len(test_data), desc="Intent Calling"):
        result = future.result()
        if result:
            test_result.append(result)
            result = json.dumps(result, ensure_ascii=False)
            fw.write(result + "\n")


# 统计指标
intent_correct = 0
slot_correct = 0
intent_and_slot_correct = 0
total = 0
intent_result = {}
for info in test_result:
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
