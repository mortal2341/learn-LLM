# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: 前处理和格式化预训练语料 
# --------------------------------------------


import os
import re
import json
import time
import unicodedata
from tqdm import tqdm
import pyarrow.parquet as pq


MIN_LEN = 10
MAX_LEN = 2500


class FormatHandler():
    def __init__(self, input_path, output_path, dataset_name):
        self.input_path = input_path
        self.output_path = output_path
        self.dataset_name = dataset_name


    def get_file_list(self) -> list:
        """根据后缀过滤文件, glob"""
        files = os.listdir(self.input_path)
        files = [i for i in files if ".json" in i or "parquet" in i]
        return files


    def process_one_line(self, idx, line):
        """处理一行数据，子类必须实现"""
        raise NotImplementedError

    
    def process_one_file(self, file_path):
        """处理一份文件"""
        line_count = 0
        jump_count = 0
        raw_data = open(self.input_path + "/" + file_path, "r", encoding="utf-8").readlines()
        line_count = len(raw_data)
        with open(self.output_path, "a") as fout:
            for idx, line in tqdm(enumerate(raw_data), total=line_count):
                idx, result = self.process_one_line(idx, line)
                if result:
                    fout.write(result)
                else:
                    jump_count += 1
        return line_count, jump_count

    
    def process_all(self):
        """处理全部文件"""
        st = time.time()
        line_count_all = 0
        jump_count_all = 0
        file_list = self.get_file_list()
        print("[log][{}] number of files is {:d}".format(self.dataset_name, len(file_list)))
        for file in file_list:
            line_count, jump_count = 0, 0
            try:
                line_count, jump_count = self.process_one_file(file)
            except Exception as e:
                print("[exception][{}] process file {} failed: {}".format(self.dataset_name, file, e))
            line_count_all += line_count
            jump_count_all += jump_count
        print("[log][{}] timecost is {:.2f} s!".format(self.dataset_name, time.time() - st))
        print("[log][{}] line_count is {:d}, jump_count is {:d}".format(self.dataset_name, line_count_all, jump_count_all))


    def length_assurance(self, line) -> bool:
        """确保一段文字的字数"""
        if len(line) < MIN_LEN or len(line) > MAX_LEN:
            return False

        return True

    def make_answer(self, reason_content, content):
        return f"<think>\n{reason_content}\n</think>\n\n<answer>\n{content}\n</answer>"


    def qa2txt(self, question: list[str], answer: list[str]):
        """将对话转换为txt文本。qwen风格"""
        if  len(question) != len(answer):
            return ""
        res = {"conversation": []}
        for i in range(len(question)):
            res["conversation"].append({"role": "user", "content": question[i]})
            res["conversation"].append({"role": "assistant", "content": answer[i]})
        return json.dumps(res, ensure_ascii=False)
        

class Reasoning110kFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(Reasoning110kFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def process_one_line(self, idx, line):
        data = json.loads(line)
        question = data["input"]
        content = data["content"].strip()
        reasoning_content = data["reasoning_content"].strip()
        answer = self.make_answer(reasoning_content, content)

        # 多轮对话转文本
        text = self.qa2txt([question], [answer])

        # 验证长度
        if not self.length_assurance(text):
            return idx, None

        return idx, text + "\n"


class DeepSeekR1ZHFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(DeepSeekR1ZHFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def process_one_line(self, idx, line):
        data = json.loads(line)["conversation"]
        if len(data) > 2:
            return idx, None

        question = [item["content"] for item in data if item["role"] == "user"][0]
        target = [item["content"] for item in data if item["role"] == "assistant"][0]
        reasoning_content, content = target.split("</think>")
        reasoning_content = reasoning_content.lstrip("<think>").strip() 
        content = content.strip()
        answer = self.make_answer(reasoning_content, content)

        # 多轮对话转文本
        text = self.qa2txt([question], [answer])

        # 验证长度
        if not self.length_assurance(text):
            return idx, None

        return idx, text + "\n"


class DistillR1FormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(DistillR1FormatHandler, self).__init__(input_path, output_path, dataset_name)


    def process_one_file(self, file_path):
        """处理一份文件"""
        line_count = 0
        jump_count = 0
        raw_data = json.load(open(self.input_path + "/" + file_path))
        line_count = len(raw_data)
        with open(self.output_path, "a") as fout:
            for idx, line in tqdm(enumerate(raw_data), total=line_count):
                idx, result = self.process_one_line(idx, line)
                if result:
                    fout.write(result)
                else:
                    jump_count += 1
        return line_count, jump_count


    def process_one_line(self, idx, line):
        question = line["instruction"]
        output = line["output"]
        reasoning_content = re.findall(r'<think>(.*?)</think>', output, re.DOTALL)[0].strip() 
        content = re.findall(r'<answer>(.*?)</answer>', output, re.DOTALL)[0].strip()
        if len(reasoning_content) < MIN_LEN or len(content) < MIN_LEN:
            return idx, None

        answer = self.make_answer(reasoning_content, content)

        # 多轮对话转文本
        text = self.qa2txt([question], [answer])

        # 验证长度
        if not self.length_assurance(text):
            return idx, None

        return idx, text + "\n"


def main_run():
    input_path_root = "think_raw/"
    output_path_root = "formatted_data_think_2048/"
    if not os.path.exists(output_path_root):
        os.makedirs(output_path_root)

    dataset_process_info = {
        "reasoning_110k": (input_path_root + "/reasoning_110k", Reasoning110kFormatHandler),
        "deepseek_r1_zh": (input_path_root + "/deepseek_r1_zh", DeepSeekR1ZHFormatHandler),
        "distill_r1_zh": (input_path_root + "/distill_r1_zh", DistillR1FormatHandler),
    }

    for dataset_name, info in dataset_process_info.items():
        input_path, Handler = info
        output_path = output_path_root + "/processed_{}.jsonl".format(dataset_name)
        if os.path.exists(output_path):
            os.remove(output_path)
        fh = Handler(input_path, output_path, dataset_name)
        fh.process_all()


if __name__ == "__main__":
    main_run()
