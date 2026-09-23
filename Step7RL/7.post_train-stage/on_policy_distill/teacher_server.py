# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: 教师模型server 
# --------------------------------------------


import torch
import logging
import json, os
import torch.nn as nn
import bottle, threading, queue
from bottle import request
from transformers import AutoModelForCausalLM
from utils import * 


# 支持tokenize并行，加速
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


# 加载教师模型
teacher_model = AutoModelForCausalLM.from_pretrained(config["model"]["teacher_model_path"],
        torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
teacher_model.eval()
teacher_model.requires_grad_(False)


# 初始化队列（last-in-first-out）
request_queue = queue.LifoQueue() # 请求队列
result_queue = queue.LifoQueue() # 结果队列

# 实例化app
app = bottle.Bottle()

def get_per_token_logps(input_ids):
    """计算reference model log probabilities"""
    logits = teacher_model(input_ids).logits  # (batch, seq_len, vocab)
    logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab)
    input_ids = input_ids[:, 1:]  # (batch, seq_len-1)
    per_token_logps = []

    # 把logits 转换为 log-p (使用循环减少峰值显存占用)
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
        
    return torch.stack(per_token_logps) # (batch, seq_len-1)


@app.route('/upload', method='POST')
def do_upload():
    raw_bytes = request.body.read()
    raw = bytes_list_to_list(raw_bytes)

    # 检查入参个数
    #if len(raw) != 4:
    if len(raw) not in (3,4):
        return b'tensor'

    # cong byte解码回数据，并转成tensor
    data = {'base': json.loads(raw[0])} # prompt length
    data['inputs'] = bytes_to_tensor(raw[1]) # input ids
    data['rewards'] = bytes_to_tensor(raw[2]) # rewards
    data['gen_logps'] = bytes_to_tensor(raw[3]) # 采样模型的logp

    # 放入请求队列
    request_queue.put(data)

    logger.info(f"recieve data: inputs: {data['inputs'].shape}, rewards: {data['rewards']}, gen_logps: {data['gen_logps'].shape}")

    return b'tensor'


@app.route('/get', method='GET')
def do_get():
    if result_queue.empty():
        return b'empty'

    return result_queue.get()


def run_server():
    bottle.run(app, host='0.0.0.0', port=int(config["server"]["port"]), server='tornado')

threading.Thread(target=run_server, daemon=False).start()

while True:
    d = request_queue.get()
    prompt_length = d['base']['prompt_len']

    # 计算参考模型的logps
    with torch.inference_mode():
        per_token_logps = get_per_token_logps(d['inputs'].to(teacher_model.device))
    per_token_logps = per_token_logps[:,prompt_length-1:] # 截取回答部分
    data = [
        json.dumps(d['base']).encode(),
        tensor_to_bytes(d['inputs']), 
        tensor_to_bytes(d['rewards']),
        tensor_to_bytes(per_token_logps),
        tensor_to_bytes(d['gen_logps'])
    ]
    bytes_data = make_bytes_list(data)
    result_queue.put(bytes_data)
    logger.info(f"response data: ref_logps: {per_token_logps.shape}")


