CUDA_VISIBLE_DEVICES=0,1 vllm serve "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
--host 0.0.0.0 --port 8000 --tensor-parallel-size 2 --gpu-memory-utilization 0.8 --served-model-name "Qwen32B"
