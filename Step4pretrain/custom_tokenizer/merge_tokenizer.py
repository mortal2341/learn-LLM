# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: 合并新增词表并验证
# --------------------------------------------


import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer


def decode_to_chinese(input_str):
    # 将unicode字符解码为中文

    unicode_points = [ord(char) for char in input_str]
    decoded_points = [point - 162 if point > 255 else point for point in unicode_points]

    try:
        byte_array = bytes(decoded_points)
        decoded_string = byte_array.decode('utf-8')
        return decoded_string
    except Exception as e:
        return input_str


def combine_tokenizers(src_tokenizer, new_tokenizer, tgt_tokenizer):

    # 加载词表文件 
    src_vocab = json.load(open(os.path.join(src_tokenizer, 'vocab.json')))
    new_vocab = json.load(open(os.path.join(new_tokenizer, 'vocab.json')))

    # 合并vocab.json 
    tgt_vocab = {}
    idx = 0
    for word in new_vocab.keys():
        if word not in tgt_vocab.keys():
            tgt_vocab[word] = idx
            idx += 1
    for word in src_vocab.keys():
        if word not in tgt_vocab.keys():
            tgt_vocab[word] = idx
            idx += 1

    if not os.path.exists(tgt_tokenizer):
        os.makedirs(tgt_tokenizer)

    # 合并merges.txt
    tgt_merges = []
    with open(os.path.join(src_tokenizer, 'merges.txt')) as fd:
        merges = fd.readlines()
        header = merges[0] 
        tgt_merges.append(header)
        for line in tqdm(merges[1:], total=len(merges)-1):
            tgt_merges.append(line.strip())

    with open(os.path.join(new_tokenizer, 'merges.txt')) as fd:
        merges = fd.readlines()
        for line in tqdm(merges[1:], total=len(merges)-1):
            line = line.strip()
            if line not in tgt_merges:
                tgt_merges.append(line)

    # 合并tokenizer.json
    tgt_token = json.load(open(os.path.join(new_tokenizer, 'tokenizer.json')))
    tgt_token["model"]["vocab"] = tgt_vocab
    tgt_merges_list = [k.split(" ") for k in tgt_merges[1:]] 
    tgt_token["model"]["merges"] = tgt_merges_list

    # 保存vocab.json
    with open(os.path.join(tgt_tokenizer, 'vocab.json'), 'w') as fp:
        json.dump(tgt_vocab, fp, ensure_ascii=False)

    # 保存merges.txt
    with open(os.path.join(tgt_tokenizer, 'merges.txt'), 'w') as fp:
        for line in tgt_merges:
            fp.write(line + "\n")

    # 保存tokenizer.json
    with open(os.path.join(tgt_tokenizer, 'tokenizer.json'), 'w') as fp:
        json.dump(tgt_token, fp, ensure_ascii=False)

    # 保存tokenizer_config.json
    os.system('cp {} {}'.format(os.path.join(new_tokenizer, 'tokenizer_config.json'), tgt_tokenizer))

    # 验证合并后的tokenizer 
    src_t = AutoTokenizer.from_pretrained(src_tokenizer)
    new_t = AutoTokenizer.from_pretrained(new_tokenizer)
    tgt_t = AutoTokenizer.from_pretrained(tgt_tokenizer)

    test_input = "天气怎么样, How's the weather?"
    print("输入：", test_input)
    src_t = src_t.tokenize(test_input)
    new_t = new_t.tokenize(test_input)
    tgt_t = tgt_t.tokenize(test_input)
    src_t = [decode_to_chinese(k) for k in src_t]
    new_t = [decode_to_chinese(k) for k in new_t]
    tgt_t = [decode_to_chinese(k) for k in tgt_t]
    print(f"{src_tokenizer} 编码: ", src_t, "编码长度: ", len(src_t))
    print(f"{new_tokenizer} 编码: ", new_t, "编码长度: ", len(new_t))
    print(f"{tgt_tokenizer} 编码: ", tgt_t, "编码长度: ", len(tgt_t))



def main():
    src_tokenizer = "./tokenizers/gpt2_tokenizer"
    new_tokenizer = "./tokenizers/custom_tokenizer"
    tgt_tokenizer = "./tokenizers/merged_tokenizer"

    combine_tokenizers(src_tokenizer, new_tokenizer, tgt_tokenizer)

if __name__ == '__main__':
    main()
