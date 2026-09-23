# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: Swift DPO训练数据处理代码
# --------------------------------------------


import json
from prompts import PROMPT

# 处理训练数据
fd = open("train_dpo_final_5k.json")
fw = open("train_dpo_final_5k_for_swift.json", "w")
data = []
for line in fd:
    line = json.loads(line)
    template = {
        "messages": [
            {
                "role": "system", 
                "content": PROMPT
            }, 
            {
                "role": "user", 
                "content": line["prompt"]
            }, 
            {
                "role": "assistant", 
                "content": line["chosen"]
            }
        ], 
        "rejected_response": line["rejected"]
    }
    data.append(template)
    fw.write(json.dumps(template, ensure_ascii=False) + "\n")
    
print("train_size:", len(data))
data_str = json.dumps(data, ensure_ascii=False, indent=2) 
fw.write(data_str)
