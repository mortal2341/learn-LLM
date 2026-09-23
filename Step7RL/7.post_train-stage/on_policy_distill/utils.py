# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: 参考模型server 
# --------------------------------------------

import io
import re
import yaml
import torch
import logging
from math_verify import parse, verify, ExprExtractionConfig

logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
                    datefmt='## %Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()


# 加载配置文件
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


def tensor_to_bytes(t):
    """从tensor转换为bytes"""
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()

def bytes_to_tensor(b):
    """从bytes转换为tensor"""
    return torch.load(io.BytesIO(b), weights_only=True)

def make_bytes_list(bytes_list):
    """从bytes转换为list[int]"""
    buffer = io.BytesIO()
	# 先写入参数个数
    buffer.write(len(bytes_list).to_bytes(4, 'big'))
    for b in bytes_list:
	    # 再写入每个参数的bytes数
        buffer.write(len(b).to_bytes(4, 'big'))
		# 最后写入实际bytes数据
        buffer.write(b)
    return buffer.getvalue()

def bytes_list_to_list(b):
    """从bytes转换为list[data]"""
    buffer = io.BytesIO(b)
    # 先解码参数个数
    param_num = int.from_bytes(buffer.read(4), 'big')
    bytes_list = []
    for _ in range(param_num):
	    # 再解码每个参数的bytes数
        l = int.from_bytes(buffer.read(4), 'big')
		# 最后解码实际的数据
        bytes_list.append(buffer.read(l))
    return bytes_list

def reward_correct(item, answer):
    """回答正确性得分"""
    answer_regex = r"<answer>(.*?)<\/answer>"
    answer_match = re.search(answer_regex, answer, re.DOTALL)
    if not answer_match:
        return 0.0

    answer_content = answer_match.group(1)
    if not answer_content:
        return 0.0

    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer_content) 
    if len(nums) == 0:
        return -1.0

    lastnum = nums[-1].strip()
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    if verify(ans, ground_truth):
        return 1 
    else:
        return -1
    
def reward_format(item, answer):
    """回答格式得分"""
    pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
    think_count = answer.count("<think>") + answer.count("</think>")
    answer_count = answer.count("<answer>") + answer.count("</answer>")
    if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count==2 and answer_count==2:
        return 1.25
    else:
        return -1

def reward_length(answer_id, l_max, l_cache):
    """回答长度得分"""
    ans_len = len(answer_id)
    if ans_len < l_max - l_cache:
        return 0
    elif ans_len > l_max:
        return -1
    else:
        return (l_max - l_cache - ans_len) / l_cache
