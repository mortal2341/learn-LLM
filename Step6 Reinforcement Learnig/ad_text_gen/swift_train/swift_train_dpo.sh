# DPO运行脚本

CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --model_type phi4 \
    --rlhf_type dpo \
    --model /root/autodl-tmp/ad_text_gen/output/phi-4-sft/merged_phi4_sft/ \
    --train_type lora \
    --dataset data_process/train_dpo_final_5k_for_swift.json \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-6 \
    --lora_rank 32 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --rpo_alpha 0.1 \
    --dataset_num_proc 8

