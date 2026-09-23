solver_model_path=$1
questioner_model_path=$2
save_path=$3

echo $STORAGE_PATH

echo "start train questioner $questioner_model_path $save_path" 

bash vllm_service_init/start.sh $solver_model_path &


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./examples/reward_function/caller.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=4 \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=16 \
    worker.actor.global_batch_size=4 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    trainer.max_steps=11

# python gpu_burn.py

pkill python    

sleep 1

python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_10/actor