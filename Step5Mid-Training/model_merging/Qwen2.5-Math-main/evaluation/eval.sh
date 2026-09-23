set -ex

export CUDA_VISIBLE_DEVICES=0,1

PROMPT_TYPE=qwen25-math-cot
MODEL_NAME_OR_PATH=../../models/merged_qwen_math_7B
# MODEL_NAME_OR_PATH=../../models/Qwen/Qwen2___5-Math-7B
# MODEL_NAME_OR_PATH=../../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
MAX_TOKEN=10240
NUM_SHOTS=0
DATASETS=gsm8k
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME=${DATASETS}
TOKENIZERS_PARALLELISM=false \

python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --max_tokens_per_call ${MAX_TOKEN} \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --num_shots ${NUM_SHOTS} \
    --save_outputs \
    --overwrite \

