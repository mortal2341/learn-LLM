CUDA_VISIBLE_DEVICES=0 vllm serve "LLM-Research/Phi-4-mini-instruct/"  --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.7 --served-model-name "phi-4-mini"

