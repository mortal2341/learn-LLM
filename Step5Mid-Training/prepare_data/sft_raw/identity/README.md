---
dataset_info:
  features:
  - name: instruction
    dtype: string
  - name: response
    dtype: string
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  splits:
  - name: train_sft
    num_bytes: 156396
    num_examples: 728
  - name: test_sft
    num_bytes: 8627
    num_examples: 39
  download_size: 19426
  dataset_size: 165023
configs:
- config_name: default
  data_files:
  - split: train_sft
    path: data/train_sft-*
  - split: test_sft
    path: data/test_sft-*
---
Datasets of basic instructions and answers for [SmolLM-Instruct](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966) models trainings: it includes answers to greetings and questions such as "Who are you". This dataset was included in training of SmolLM-Instruct v0.2 but we didn't notice that it had an impact on model generations. 
We recommend using this generic larger dataset of multi-turn everyday conversations: https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k

