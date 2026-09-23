# 处理医疗领域数据，领域：通用按1:1混合

import json
import random

MIN_LEN = 20
MAX_LEN = 768
N_GENRAL = 290000

data = []
fd = open("train_encyclopedia.json", "r", encoding="utf-8")
for line in fd:
    info = json.loads(line)
    if len(info["text"]) > MIN_LEN and len(info["text"]) < MAX_LEN:
        data.append(info)

n_medical_samples = len(data)
print("medical samples: ", n_medical_samples)

fd = open("processed_dataset_512_clean_230w.jsonl", "r", encoding="utf-8")
for idx, line in enumerate(fd):
    if idx <= N_GENRAL:
        data.append(json.loads(line))

print("general samples: ", len(data) - n_medical_samples)
print("total samples: ", len(data))


random.shuffle(data)
fw = open("processed_medical_mix.jsonl", "w", encoding="utf-8")
for info in data:
    fw.write(json.dumps(info, ensure_ascii=False) + "\n")


