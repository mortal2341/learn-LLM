# sudo apt-get install jq

jq -c 'select(.pos_type == "拒绝&正向建议" and .neg_type == "风险回复")' data/iic/CValues-Comparison/train.jsonl > data/dpo_train.jsonl
jq -c 'select(.pos_type == "拒绝&正向建议" and .neg_type == "风险回复")' data/iic/CValues-Comparison/test.jsonl > data/dpo_test.jsonl
