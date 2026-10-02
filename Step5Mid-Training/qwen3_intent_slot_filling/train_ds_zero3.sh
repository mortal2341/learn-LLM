#!/usr/bin/env bash
# ---------------------------------------------------
# ZeRO-3 训练：把冻结的 8B 基座权重也做分片
# 单卡权重 16.38GB -> 4.1GB，是本项目唯一有意义的显存优化
# ---------------------------------------------------
set -e
export DS_CONFIG=configs/ds_zero3.json
deepspeed --num_gpus 4 train_ner.py
