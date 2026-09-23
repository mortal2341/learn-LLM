import json
from prompts import PROMPT

# 处理训练数据
fd = open("train_dpo_final_5k.json")
fw_p = open("train_dpo_final_5k_for_llama.json", "w")
fw_s = open("train_sft_final_5k_for_llama.json", "w")
dpo_data = []
sft_data = []
for line in fd:
    line = json.loads(line)
    dpo_template = {
        "system": PROMPT,
        "instruction": "",
        "input": line["prompt"],
        "chosen": line["chosen"],
        "rejected": line["rejected"]
    }
    dpo_data.append(dpo_template)

    sft_template = {
        "instruction": PROMPT,
        "input": line["prompt"],
        "output": line["chosen"],
    }
    sft_data.append(sft_template)

print("train_size:", len(dpo_data))
dpo_data_str = json.dumps(dpo_data, ensure_ascii=False, indent=2)
sft_data_str = json.dumps(sft_data, ensure_ascii=False, indent=2)
fw_p.write(dpo_data_str)
fw_s.write(sft_data_str)
