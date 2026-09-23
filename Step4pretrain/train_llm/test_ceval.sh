lm_eval --model hf \
    --model_args pretrained=saved/pretrain/pretrain-qwen-0.1B/ \
    --tasks ceval* \
    --device cuda \
    --batch_size 24 \
    --trust_remote_code
