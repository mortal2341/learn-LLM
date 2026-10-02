#!/usr/bin/env bash
# ---------------------------------------------------
# ZeRO-2 训练：分片优化器状态 + 梯度
# 注意：本项目是 LoRA(r=8)，可训练参数只有 21.8M，
#       ZeRO-1/2 能省的显存 < 0.3GB/卡，基本可忽略。
#       想要显存收益请用 train_ds_zero3.sh
# ---------------------------------------------------
set -e
export DS_CONFIG=configs/ds_zero2.json
deepspeed --num_gpus 4 train_ner.py
