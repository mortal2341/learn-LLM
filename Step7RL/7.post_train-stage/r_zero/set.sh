export STORAGE_PATH=./saved/ckpt
export HUGGINGFACENAME=xxx
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/post_train_stage/r_zero

mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"
