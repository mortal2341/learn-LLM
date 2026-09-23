shuf data/processed_dataset_512.jsonl | sed 's/<|im_start|>//g' | sed 's/<|im_end|>//g' > data/processed_dataset_512_clean_230w.jsonl 
