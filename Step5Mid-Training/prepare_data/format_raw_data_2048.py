# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: 中期训练前处理和格式化预训练语料 
# --------------------------------------------


import os
import re
import json
import time
import unicodedata
from tqdm import tqdm
import pyarrow.parquet as pq


MIN_LEN = 768
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
        with open(self.output_path, "a",encoding="utf-8") as fout:
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

    def zh_process(self, line) -> str:
        """初步的中文文本处理"""
        # 1. None 处理成空字符串
        if line is None:
            return ""

        # 2. unicode 统一
        line = unicodedata.normalize("NFKC", line)

        # 3. 去除引用，针对维基百科[1]
        line = re.sub(r'(\[\d+\]\s*)+$', '', line).rstrip()

        return line

    
    def qa2txt(self, question: str, answer: str):
        """将对话转换为txt文本, qwen风格"""
        return "<|im_start|>{}<|im_end|>".format(question) + \
            "<|im_start|>{}<|im_end|>".format(answer)
        

class BELLEFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(BELLEFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def process_one_line(self, idx, line):
        data = json.loads(line)
        if "instruction" not in data:
            return idx, None

        instruction = self.zh_process(data["instruction"])
        output = self.zh_process(data["output"])
        text = self.qa2txt(instruction, output)

        if not self.length_assurance(text):
            return idx, None

        return idx, json.dumps({"text": text}, ensure_ascii=False) + "\n"


class FireflyFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(FireflyFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def process_one_line(self, idx, line):
        data = json.loads(line)
        input_str = self.zh_process(data["input"])
        target_str = self.zh_process(data["target"])
        text = self.qa2txt(input_str, target_str)

        if not self.length_assurance(text):
            return idx, None

        return idx, json.dumps({"text": text}, ensure_ascii=False) + "\n"


class WikiCNFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(WikiCNFormatHandler, self).__init__(input_path, output_path, dataset_name)
    
    def process_one_line(self, idx, line):
        line = json.loads(line)
        item = line["text"]
        text = self.zh_process(item)
        text = text[:MAX_LEN]
        if not self.length_assurance(text):
            return idx, None

        return idx, json.dumps({"text": text}, ensure_ascii=False) + "\n"


class Reason110KFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(Reason110KFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def process_one_line(self, idx, line):
        data = json.loads(line)
        input_str = self.zh_process(data["input"])
        content_str = self.zh_process(data["content"])
        reasoning_content_str = self.zh_process(data["reasoning_content"])
        target_str = reasoning_content_str + content_str
        text = self.qa2txt(input_str, target_str)

        if not self.length_assurance(text):
            return idx, None

        return idx, json.dumps({"text": text}, ensure_ascii=False) + "\n"


class MathCodeFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(MathCodeFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def process_one_file(self, file_path):
        """处理一份文件"""
        line_count = 0
        jump_count = 0
        table = pq.read_table(self.input_path + "/" + file_path)
        raw_data = table.to_pydict()
        line_count = len(raw_data["conversations"])
        with open(self.output_path, "a",encoding="utf-8") as fout:
            for idx, line in tqdm(enumerate(raw_data["conversations"]), total=line_count):
                idx, result = self.process_one_line(idx, line)
                if result:
                    fout.write(result)
                else:
                    jump_count += 1
        return line_count, jump_count

    def process_one_line(self, idx, line):
        text = "\n".join([item["value"] for item in line])
        if not self.length_assurance(text):
            return idx, None

        return idx, json.dumps({"text": text}, ensure_ascii=False) + "\n"


class SynthesisTextBooksFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(SynthesisTextBooksFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def process_one_file(self, file_path):
        """处理一份文件"""
        line_count = 0
        jump_count = 0
        table = pq.read_table(self.input_path + "/" + file_path)
        raw_data = table.to_pydict()
        line_count = len(raw_data["text"])
        with open(self.output_path, "a", encoding="utf-8") as fout:
            for idx, line in tqdm(enumerate(raw_data["text"]), total=line_count):
                idx, result = self.process_one_line(idx, line)
                if result:
                    fout.write(result)
                else:
                    jump_count += 1
        return line_count, jump_count

    def process_one_line(self, idx, line):
        if not self.length_assurance(line):
            return idx, None

        return idx, json.dumps({"text": line}, ensure_ascii=False) + "\n"




def main_run():
    input_path_root = "raw_data/"
    output_path_root = "formatted_data_2048/"
    if not os.path.exists(output_path_root):
        os.makedirs(output_path_root)

    dataset_process_info = {
        "wiki_baike": (input_path_root + "/wikibaike", WikiCNFormatHandler),
        "BELLE": (input_path_root + "/belle", BELLEFormatHandler),
        "firefly": (input_path_root + "/firefly", FireflyFormatHandler),
        "reasoning": (input_path_root + "/reasoning_110k", Reason110KFormatHandler),
        "code_math": (input_path_root + "/code_math", MathCodeFormatHandler),
        "textbook": (input_path_root + "/text_books", SynthesisTextBooksFormatHandler),
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
