# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: GRPO在线强化蒸馏训练
# --------------------------------------------


import torch
import copy
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import json, os, re, random, io, requests, sys, time
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from prompts import SYSTEM_PROMPT
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from utils import *

print("Training settings:")
print(config)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
ref_server = f'{config["server"]["ip"]}:{config["server"]["port"]}'

# 加载tokenizere
tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_path"])


def get_batch():
    "从参考模型的server返回里取出一个batch数据"
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty':
            return None
    except:
        return None
        
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['teacher_logps'] = bytes_to_tensor(dd[3])
    data['gen_logps'] = bytes_to_tensor(dd[4])
    return data

def get_per_token_logps(logits, input_ids):
    """计算策略模型的log-p"""
    per_token_logps = [] # 用循环计算 logsoftmax 减少 memory peak
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def GRPO_step(batch):
    """GRPO 模型更新"""

    prompt_length = batch['prompt_len']
    inputs = batch['inputs'].to(engine.device)# (batch, seq_len)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)  # (batch, 1)
    logits = engine(inputs).logits # (batch, seq_len, vocab)
    logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab)
    input_ids = inputs[:, 1:]  # (batch, seq_len-1)
    per_token_logps = get_per_token_logps(logits, input_ids) # (batch, seq_len-1, vocab)
    per_token_logps = per_token_logps[:,prompt_length-1:] # (batch, answer_len)
    old_per_token_lops = batch['gen_logps'].to(engine.device) # (batch, answer_len)
    teacher_per_token_logps = batch['teacher_logps'].to(per_token_logps.device) # (batch, answer_len)

    # 输出batch answer的mask
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int() # (batch, answer_len)

    # log(p) - log(old_p)
    ratio = torch.exp(per_token_logps - old_per_token_lops) # (batch, answer_len)

    # log(p) - log(teacher_p), teacher model的reversed kl作为dense reward
    reversed_kl = torch.exp(per_token_logps - teacher_per_token_logps) # (batch, answer_len)
    reversed_kl = torch.clamp(reversed_kl, -config["training"]["max_kl"], config["training"]["max_kl"])
    
    # grpo clip ratio
    clipped_ratio = torch.clamp(ratio, 1-config["training"]["clip_param"], 1+config["training"]["clip_param"]) # (batch, answer_len)

    # 合并RLVR和reversed_kl
    combined_advantages = -config["training"]["kl_penalty_coef"]*reversed_kl + advantages
    
    # -min(ratio*A, clip(ratio)*A)
    per_token_loss = -torch.min(ratio * combined_advantages, clipped_ratio * combined_advantages) # (batch, answer_len)
    
    # 取样本平均 
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    return loss


def gen_worker(model_queue, physics_device):
    """数据生成管道"""
    
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" # 禁用v1的多进程，代码的vllm版本是0.11
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}' # 采样gpu
    cleanup_keys = [  
            'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'LOCAL_RANK',  
            'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_NAME',   
            'GROUP_WORLD_SIZE', 'ROLE_WORLD_SIZE',  
            'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_MAX_RESTARTS',  
            'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_USE_AGENT_STORE',  
            'TORCHELASTIC_ERROR_FILE',  
            'TORCH_NCCL_ASYNC_ERROR_HANDLING',  
            'NCCL_COMM_ID', 'NCCL_DEBUG', 'NCCL_SOCKET_IFNAME',  
        ]  

    # 清空以上环境变量
    for key in cleanup_keys:
        os.environ.pop(key, None)
        
    torch.cuda.set_device(0)   # 设置当前GPU设备编号为0
    logger.info(f"Generation worker process uses GPU {physics_device}")
    
    # 启动本地vllm
    vllm_gen = LLM(
        model=config["model"]["model_path"],
        gpu_memory_utilization=config["gpu"]["gpu_memory_utilization"])
    
    ref_server_ver = 'tensor'  # 标识位，用于请求参考模型的server，string和tensor标识位轮换
    Q_batch_size = config["training"]["Q_batch_size"]  # 一次rollout的prompt数
    num_pre_Q = config["training"]["num_pre_Q"]  # 每一条prompt采样的答案数（group size）
    train_batch_size = config["training"]["train_batch_size"] # grpo训练的batch size

    # 采样模型参数（rollout）
    sampling_params = SamplingParams(
        n=num_pre_Q,
        temperature=config["sampling"]["temperature"],
        max_tokens=config["sampling"]["max_tokens"])
    
    # 生成模型的参数（用于计算logp，不需要随机）
    gen_sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    # 评估模型的参数
    evaluate_sampling_params = SamplingParams(n=1, temperature=0.01, max_tokens=1024)

    # 加载训练和测试数据
    dataset = load_dataset(config["data"]["data_path"], "main", split="train")
    test_dataset = load_dataset(config["data"]["data_path"], "main", split="test")

    # 提取 prompt 和 answer
    QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
    test_QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(test_dataset['question'], test_dataset['answer'])]

    # 保存推理数据
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    reasoning_log_handler = open(config["checkpoint"]["ckpt_dir"] + f"/reasoning_log_{timestamp}.txt", "w")
    
    def gen_answers(prompts, test=False):
        """调用vllm采样模型做rollout采样"""

        tip_text = []
        for prompt in prompts:
            tip_text.append(tokenizer.apply_chat_template([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True)
            )
        prompts_text = copy.deepcopy(tip_text)

        # 生成采样结果
        if test:
            voutputs = vllm_gen.generate(tip_text, evaluate_sampling_params, use_tqdm=False)
        else:
            voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)

        # 提取answer和token_ids并返回
        answers = []
        ans_token_ids = []
        for v in voutputs:
            for z in v.outputs: 
                answers.append(z.text)
                ans_token_ids.append(z.token_ids)
        
        return prompts_text, answers, ans_token_ids

    def gen_samples(inputs, test=False):
        """执行rollout获取训练样本"""

        prompts = [x["Q"] for x in inputs]

        # rollout拿到生成的answer
        prompts_text, answers, ans_token_ids = gen_answers(prompts, test)

        # 根据生成的answer计算rewards
        rewards = []
        rewards_info = []
        if test:
            for inp, answer in zip(inputs, answers):
                correct_score = reward_correct(inp, answer)
                format_score = reward_format(inp, answer)
                reward = correct_score + format_score
                rewards.append(reward)
                rewards_info.append({
                    "correct_score": correct_score,
                    "format_score": format_score,
                })
        else:
            for i, inp in enumerate(inputs):
                for answer in answers[i*num_pre_Q: (i+1)*num_pre_Q]:
                    correct_score = reward_correct(inp, answer)
                    format_score = reward_format(inp, answer)
                    reward = correct_score + format_score
                    rewards.append(reward)
                    rewards_info.append({
                        "correct_score": correct_score,
                        "format_score": format_score,
                    })

        rewards = torch.tensor(rewards, dtype=torch.float32) # 注意，这里转fp32，保证计算精度

        # prompts_text shape: Q_batch_size
        # rewards shape: Q_batch_size * num_pre_Q
        # answers shape: Q_batch_size * num_pre_Q
        # ans_token_ids shape: Q_batch_size * num_pre_Q
        return prompts_text, rewards, answers, ans_token_ids, rewards_info

    
    def try_update_model():
        """更新采样模型"""
        try:
            new_state_dict = model_queue.get_nowait()
            logger.info('[VLLM PROCESS] recving new model ...')
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            logger.info('[VLLM PROCESS] model updated')
            del new_state_dict
        except:
            return
        
    
    for it in range(10000000):
        if it % config["training"]["model_update_iter"] == 0:
            # 更新 rollout采样模型
            try_update_model()

        if (it+1) % config["training"]["eval_steps"] == 0:
            format_success_num = 0
            answer_success_num = 0
            test_batch_size = config["training"]["eval_batch_size"]
            test_size = config["training"]["eval_size"]
            eval_start = time.time()
            for i in range(0, test_size, test_batch_size):
                test_inputs = test_QAs[i:i + test_batch_size]
                _, _, _, _, rewards_info = gen_samples(test_inputs, test=True)
                
                # 评估准确率
                for info in rewards_info:
                    if np.abs(info["format_score"] - 1.25) < 1e-3:
                        format_success_num = format_success_num + 1
                    if np.abs(info["correct_score"] - 1.0) < 1e-3:
                        answer_success_num = answer_success_num + 1
                        
            format_accuracy = format_success_num / test_size
            answer_accuracy = answer_success_num / test_size
            logger.info(f"[Evaluation] iteration {it}, format accuracy: {format_accuracy}, answer accuracy: {answer_accuracy}, eval time: {time.time() - eval_start:.2f}")
        
        # 从训练数据里随机sample一个batch
        inputs = random.sample(QAs, Q_batch_size)
        
        tic = time.time()
        # 采样得到训练样本
        prompt_inputs, rewards, answers, ans_token_ids, rewards_info = gen_samples(inputs)
        logger.info(f"time: {time.time()-tic:.2f}s  iteration: {it}, rewards: {rewards}")
        if it % config["training"]["log_steps"] == 0:
            logger.info(f"prompt: {prompt_inputs[0]}, answers: {answers[0]}")

        # 保存推理采样数据（用于分析badcase）
        for inp, answer, reward in zip(inputs, answers, rewards_info):
            dump_info = {"Q": inp["Q"], "A": answer}
            reasoning_log_handler.write(json.dumps(dump_info, ensure_ascii=False) + "\n")

        # 开始计算buffer需要的数据(按batch发送给ref model server)
        for i, prompt in enumerate(prompt_inputs):
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
            prompt_len = prompt_ids.shape[1]
            curr_answers = answers[i*num_pre_Q: (i+1)*num_pre_Q]       # group size：num_pre_Q
            curr_ans_ids = ans_token_ids[i*num_pre_Q: (i+1)*num_pre_Q] # group size：num_pre_Q
            curr_rewards = rewards[i*num_pre_Q: (i+1)*num_pre_Q]       # group size：num_pre_Q
            
            # 如果一个batch内答案的reward没有区分度，则继续采样（例如都是正样本或者都是负样本）
            if curr_rewards.max() - curr_rewards.min() < 1e-4:
                continue
            

            if ref_server_ver == 'tensor':
                # 组内计算reward
                curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4) # 计算组内优势
                for ii in range(0, num_pre_Q, train_batch_size): # minibatch_size，训练批量大小
                    sub_rewards = curr_rewards[ii:ii+train_batch_size] # train_batch_size
                    sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]  # train_batch_size
                    tensor_list = [torch.tensor(lst) for lst in sub_ans_ids] # train_batch_size
                    output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id) 
                    Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, prompt_len) # 同一条prompt复制batch size份
                    merged_ids = torch.cat([Qrep, output_ids], dim=1) # 把prompt和生成的answer拼接到一起

                    # 准备打包给参考模型计算ref log-p
                    data = [
                        json.dumps({"prompt_len": prompt_len}).encode(),
                        tensor_to_bytes(merged_ids),
                        tensor_to_bytes(sub_rewards)
                    ]       

                    # 计算采样模型的log-p
                    batched_ids = merged_ids.tolist()
                    # 注意新版vllm这里要传入list[TokensPrompt]
                    prompt_token_ids = [TokensPrompt(prompt_token_ids=ids) for ids in batched_ids] 
                    gen_ids = vllm_gen.generate(prompt_token_ids, sampling_params=gen_sampling_params, use_tqdm=False)
                    # 取出answer部分的ids
                    gen_ids = [ids.prompt_logprobs[prompt_len:] for ids in gen_ids]
                    # 转成tensor
                    gen_logps = torch.tensor([[list(x.values())[0].logprob for x in ids] for ids in gen_ids])
                    data.append(tensor_to_bytes(gen_logps))

                    # 把json转换成bytes
                    xdata = make_bytes_list(data)

                    # 发给参考模型的服务
                    r = requests.post(f"{ref_server}/upload", data=xdata)
                    
                    if r.content == b'string': 
                        ref_server_ver = 'string'
                        
            elif ref_server_ver == 'string':
                # 实现tensor 和 string轮换，这里如果是string，先发一个dummy data过去
                xdata = make_bytes_list([json.dumps({"Q": prompt[0], "As": curr_answers}).encode(), 
                                        tensor_to_bytes(curr_rewards)])
                r = requests.post(f"{ref_server}/upload", data=xdata)
                if r.content == b'tensor':
                    ref_server_ver = 'tensor'

        # 刷新日志数据
        reasoning_log_handler.flush()


if __name__ == '__main__':

    # 初始化deepspeed
    import deepspeed
    deepspeed.init_distributed()

    if dist.get_rank() == 0:
        logger.info('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        model_queue = mp.Queue()
        p = mp.Process(target=gen_worker, args=(model_queue, config["gpu"]["gen_device"]))
        p.start()

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(config["model"]["model_path"], 
            dtype=torch.bfloat16, _attn_implementation="sdpa")

    # 初始化deepspeed引擎
    ds_config = json.load(open(config["deepspeed"]["config_path"]))
    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                                model_parameters=model.parameters())

    # 总共训练all_steps步
    progress = range(1, config["training"]["all_steps"] + 1)
    if dist.get_rank() == 0:
        progress = tqdm(progress)
    
    # 开始训练
    for step in progress:
        # 从参考模型服务里获取一个batch训练数据
        batch = get_batch()
        while batch is None:
            logger.info('waiting for batch...')
            time.sleep(1)
            batch = get_batch()

        # 更新GRPO模型
        loss = GRPO_step(batch)
        engine.backward(loss)
        engine.step()

        # 打印信息
        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")

        
        # 每gen_update_steps更新一次采样模型
        if step % config["training"]["gen_update_steps"] == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                logger.info('[TRAINING PROCESS] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                model_queue.put(state_dict)
                logger.info('[TRAINING PROCESS] send state_dict ok!')
            dist.barrier()

        # 每save_steps保存一次模型
        if step % config["training"]["save_steps"] == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                logger.info('saving model')
                save_name = config["checkpoint"]["ckpt_dir"] + f"/step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()
