# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: 训练自定义的tokenizer 
# --------------------------------------------


import os
import random
import json
import tokenizers
import transformers
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from transformers import AutoTokenizer
from stats import char_distribution

print("tokenizers version: ", tokenizers.__version__)
print("transformers version: ", transformers.__version__)

INPUT_FILE_PATH = "./data/processed_dataset_512_sample_160w.jsonl"
SAVE_DIR = "./tokenizers/custom_tokenizer/"
VOCAB_SIZE = 9600

random.seed(42)


def train_tokenizer():

    # 读取文本数据
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']

    # 初始化tokenizer, BBPE
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>", "<think>", "</think>", "<answer>", "</answer>"
    ]

    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 读取文本数据
    texts = read_texts_from_jsonl(INPUT_FILE_PATH)

    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 保存tokenizer
    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer.save(os.path.join(SAVE_DIR, "tokenizer.json"))
    tokenizer.model.save(SAVE_DIR)

    # 加载创建配置文件
    config = json.load(open("config.json"))

    # 保存配置文件
    with open(os.path.join(SAVE_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")


def eval_tokenizer():

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)

    messages = [
        {"role": "system", "content": "你是一个有用的智能助手"},
        {"role": "user", "content": '今天天气怎么样?'},
        {"role": "assistant", "content": '今天的的温度21摄氏度，局部零星小雨。'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)

    # 获取词汇表大小
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表大小：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print("IDs: ", model_inputs['input_ids'])
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('decoder和原始文本是否一致：', response == new_prompt)

    # 查看词表分布
    char_distribution(SAVE_DIR)


def main():
    train_tokenizer()
    eval_tokenizer()


if __name__ == '__main__':
    main()
