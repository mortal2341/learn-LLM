shuf data/processed_dataset_2048.jsonl | sed 's/<|im_end|>//g' | sed 's/<|im_start|>//g' > data/processed_dataset_2048_clean_64w.jsonl
head -160000 data/processed_dataset_512_clean_230w.jsonl > data/processed_dataset_512_clean_16w.jsonl
cat data/processed_dataset_2048_clean_64w.jsonl data/processed_dataset_512_clean_16w.jsonl | shuf > data/processed_mid_training_80w.jsonl
rm data/processed_dataset_2048_clean_64w.jsonl
rm data/processed_dataset_512_clean_16w.jsonl
