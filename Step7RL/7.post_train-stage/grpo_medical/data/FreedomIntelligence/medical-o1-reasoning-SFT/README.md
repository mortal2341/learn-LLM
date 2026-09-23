---
license: apache-2.0
task_categories:
- question-answering
- text-generation
language:
- en
- zh
tags:
- medical
- biology
configs:
- config_name: en
  data_files: medical_o1_sft.json
- config_name: zh
  data_files: medical_o1_sft_Chinese.json
- config_name: en_mix
  data_files: medical_o1_sft_mix.json
- config_name: zh_mix
  data_files: medical_o1_sft_mix_Chinese.json
---

## News

[2025/04/22] We split the data and kept only the medical SFT dataset (`medical_o1_sft.json`). The file `medical_o1_sft_mix.json` contains a mix of medical and general instruction data.

[2025/02/22] We released the [distilled dataset from Deepseek-R1](https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data) based on medical verifiable problems. You can use it to initialize your models with the reasoning chain from `Deepseek-R1`.

[2024/12/25] We open-sourced the medical reasoning dataset for SFT, built on medical verifiable problems and an LLM verifier.

## Introduction
This dataset is used to fine-tune HuatuoGPT-o1, a medical LLM designed for advanced medical reasoning. This dataset is constructed using GPT-4o, which searches for solutions to [verifiable medical problems](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-verifiable-problem) and validates them through a medical verifier. 


For details, see our [paper](https://arxiv.org/pdf/2412.18925) and [GitHub repository](https://github.com/FreedomIntelligence/HuatuoGPT-o1).




## Citation

If you find our data useful, please consider citing our work!
```
@misc{chen2024huatuogpto1medicalcomplexreasoning,
      title={HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs}, 
      author={Junying Chen and Zhenyang Cai and Ke Ji and Xidong Wang and Wanlong Liu and Rongsheng Wang and Jianye Hou and Benyou Wang},
      year={2024},
      eprint={2412.18925},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.18925}, 
}
```