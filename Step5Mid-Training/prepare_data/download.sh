export HF_ENDPOINT=https://hf-mirror.com 
mkdir -p raw_data

# pretrain
huggingface-cli download --repo-type dataset --resume-download pleisto/wikipedia-cn-20230720-filtered --local-dir ./raw_data/wikibaike
huggingface-cli download --repo-type dataset --resume-download fjcanyue/wikipedia-zh-cn --local-dir ./raw_data/wikibaike 
huggingface-cli download --repo-type dataset --resume-download YeungNLP/firefly-train-1.1M --local-dir ./raw_data/firefly
huggingface-cli download --repo-type dataset --resume-download BelleGroup/train_2M_CN --local-dir ./raw_data/belle

# midtrain
huggingface-cli download --repo-type dataset --resume-download Congliu/Chinese-DeepSeek-R1-Distill-data-110k --local-dir ./raw_data/reasoning_110k
https://huggingface.co/datasets/BAAI/Infinity-Instruct (train-00000-of-00015.parquet)
https://huggingface.co/datasets/opencsg/chinese-cosmopedia (00000.parquet)

# sft
huggingface-cli download --repo-type dataset --resume-download Mxode/I_Wonder_Why-Chinese --local-dir ./sft_raw/baike_instruct
huggingface-cli download --repo-type dataset --resume-download Mxode/Chinese-Instruct-Lite --local-dir ./sft_raw/chinese_instruct_lite
huggingface-cli download --repo-type dataset --resume-download HuggingFaceTB/instruct-data-basics-smollm-H4 --local-dir ./sft_raw/identity

# think
huggingface-cli download --repo-type dataset --resume-download jinliuxi/deepseek_r1_zh --local-dir ./think_raw/deepseek_r1_zh
huggingface-cli download --repo-type dataset --resume-download Congliu/Chinese-DeepSeek-R1-Distill-data-110k --local-dir ./think_raw/reasoning_110k
huggingface-cli download --repo-type dataset --resume-download shareAI/Alpaca-Distill-R1-ZH --local-dir ./think_raw/distill_r1_zh
