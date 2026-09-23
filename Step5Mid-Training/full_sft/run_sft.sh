deepspeed --num_gpus 4 sft_train.py --master_port 51336
# deepspeed --num_gpus 8 sft_plus_train.py --master_port 51336
# deepspeed --num_gpus 4 distill_r1_train.py --master_port 51336 --deepspeed_config config/ds_z2_tp.json
