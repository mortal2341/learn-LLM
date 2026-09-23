# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: 统计tokenizer的中文，英文，以及其他字符的占比 
# --------------------------------------------


from transformers import AutoTokenizer
import re


def char_distribution(tokenizer_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    vocab = tokenizer.get_vocab()

    total_tokens = len(vocab)
    chinese_tokens = 0
    english_tokens = 0
    non_chinese_or_english_tokens = 0

    def is_chinese(text):
        return all('\u4e00' <= char <= '\u9fff' for char in text)

    def is_english(text):
        return all(re.match(r"[a-zA-Z]", char) for char in text)

    for token_id in range(total_tokens):
        token = tokenizer.decode([token_id])
        if is_chinese(token):
            chinese_tokens += 1
        elif is_english(token):
            english_tokens += 1
        else:
            non_chinese_or_english_tokens += 1

    chinese_ratio = chinese_tokens / total_tokens * 100
    english_ratio = english_tokens / total_tokens * 100
    other_ratio = non_chinese_or_english_tokens / total_tokens * 100
    print(f"词表总大小: {total_tokens}")
    print(f"中文 token 数量: {chinese_tokens} 占比: {chinese_ratio:.2f}%")
    print(f"英文 token 数量: {english_tokens} 占比: {english_ratio:.2f}%")
    print(f"其他 token 数量: {non_chinese_or_english_tokens} 占比: {other_ratio:.2f}%")


if __name__ == "__main__":
    print("GPT2词表分布：")
    char_distribution("tokenizers/gpt2_tokenizer")
    print("="*100)

    print("新增词表分布：")
    char_distribution("tokenizers/custom_tokenizer")
    print("="*100)

    print("Qwen3词表分布：")
    char_distribution("tokenizers/qwen3_tokenizer")
    print("="*100)

    print("合并后词表分布：")
    char_distribution("tokenizers/merged_tokenizer")
    print("="*100)
