import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

import json
import jieba
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from multiprocessing import Pool
import os

# ================= 全局配置 =================
FILE_PATH = "C:/Users/1/Desktop/data/baike_qa/baike_qa_train.json"
NUM_PERM = 128
THRESHOLD = 0.7
MAX_WORKERS = 8  # 适当减少
BATCH_SIZE = 10000  # 真正的分批


def compute_minhash(args):
    """子进程任务：接收文本，返回 MinHash"""
    idx, text = args
    words = jieba.lcut(text)
    mh = MinHash(num_perm=NUM_PERM)
    for w in words:
        mh.update(w.encode("utf-8"))
    return idx, mh


if __name__ == '__main__':
    # --- 1. 读取数据 ---
    print("正在加载数据...")
    dialogs = []
    with open(FILE_PATH, 'r', encoding='utf-8') as fd:
        for line in tqdm(fd, desc="Load Data"):
            info = json.loads(line)
            dialogs.append(info["title"] + " " + info["answer"])

    total = len(dialogs)
    print(f"数据加载完成，总样本数: {total}")

    # --- 2. 初始化 LSH ---
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    minhashes = [None] * total

    # --- 3. 真正的分批处理（核心修复！）---
    print("正在生成 Hash 向量并插入 LSH...")

    with Pool(processes=MAX_WORKERS) as pool:
        for batch_start in tqdm(range(0, total, BATCH_SIZE), desc="Batches"):
            batch_end = min(batch_start + BATCH_SIZE, total)

            # ✅ 只传文本，不传整个 dialogs 列表
            batch_args = [(i, dialogs[i]) for i in range(batch_start, batch_end)]

            # ✅ imap_unordered：惰性迭代，处理完一个释放一个
            for idx, mh in pool.imap_unordered(compute_minhash, batch_args, chunksize=200):
                # ✅ 修复：防止重复 key 报错（容错处理）
                try:
                    lsh.insert(f"dialog_{idx}", mh)
                except ValueError:
                    pass  # 如果 key 已存在则跳过
                minhashes[idx] = mh

            # ✅ 每批处理完，batch_args 自动释放
            del batch_args

    print("Hash 生成完成！")

    # --- 4. 去重逻辑（必须缩进在 if __name__ == '__main__' 内部！）---
    print("开始去重...")

    unique_dialogs = []  # 去重后的文档
    seen = set()  # 标记已处理的文档
    print_count = 0
    max_print = 3

    # ✅ 修复：total_dialogs 改为 total
    for idx in tqdm(range(total), desc="Deduplicate docs"):
        if idx in seen:
            continue

        minhash = minhashes[idx]
        if minhash is None:
            continue

        result = lsh.query(minhash)  # 查询相似文档

        # 找当前文档的相似文档索引（排除自己）
        similar_idxs = [int(r.split("_")[1]) for r in result if r != f"dialog_{idx}"]

        if not similar_idxs:  # 没有相似的文档
            unique_dialogs.append(dialogs[idx])
        else:  # 有相似文档
            unique_dialogs.append(dialogs[idx])

            # ✅ 修复：过滤掉越界的索引，防止后续打印报错
            valid_similar_idxs = [i for i in similar_idxs if 0 <= i < total]
            seen.update(valid_similar_idxs)  # 标记相似文档为已处理

            # 打印相似对话信息
            if print_count < max_print and valid_similar_idxs:
                # ✅ 修复：threshold 改为 THRESHOLD
                print(f"\n输入： \n '{dialogs[idx]}' \n 与以下文档相似: (阈值={THRESHOLD}): \n")
                for sim_idx in valid_similar_idxs[:max_print]:
                    print(f"\n 输出 {sim_idx}: \n '{dialogs[sim_idx]}'")
                print("*" * 100)
                print_count += 1

        seen.add(idx)

    # 打印去重结果的统计信息
    print("\n==== 去重结果 ====")
    # ✅ 修复：total_dialogs 改为 total
    print(f"原始文档数量: {total}")
    print(f"去重后的文档数量: {len(unique_dialogs)}")
    print(f"重复的文档数量: {total - len(unique_dialogs)}")

    print("处理完成！")  # ✅ 移到最后一行