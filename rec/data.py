"""数据层：MovieLens-100k 下载、清洗、隐式反馈构建与留出法划分。

召回任务使用「隐式反馈」：把评分 >= min_rating 的交互视为正样本（表示用户喜欢）。
评估使用「留一法（leave-one-out）」：每个用户按时间排序，最近一次交互进测试集，其余进训练集。
"""

from __future__ import annotations

import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_ml100k(dest_dir: Path) -> Path:
    """下载并解压 MovieLens-100k 的 u.data，返回其路径（已存在则复用缓存）。"""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "ml-100k.zip"
    data_path = dest_dir / "u.data"

    if not data_path.exists():
        if not zip_path.exists():
            # 本机网络对默认证书链校验失败，关闭校验后下载（仅限公开数据集）
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(ML100K_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                zip_path.write_bytes(resp.read())
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open("ml-100k/u.data") as src:
                data_path.write_bytes(src.read())
    return data_path


def load_interactions(
    data_path: Path, min_rating: float = 4.0
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """读取 u.data -> 过滤为正样本 -> 把 user/item 重映射为从 0 开始的连续 id。"""
    df = pd.read_csv(data_path, sep="\t", names=["user", "item", "rating", "ts"])
    df = df[df["rating"] >= min_rating].copy()
    u_code, u_uniq = pd.factorize(df["user"])
    i_code, i_uniq = pd.factorize(df["item"])
    df["u"] = u_code.astype(np.int64)
    df["i"] = i_code.astype(np.int64)
    return df, u_uniq.to_numpy(), i_uniq.to_numpy()


def leave_one_out_split(df: pd.DataFrame, min_train: int = 1):
    """按时间对每个用户留一：最后一次交互进测试集。丢弃训练后为空的用户。"""
    df = df.sort_values(["u", "ts"], kind="mergesort")
    test = df.groupby("u", sort=False).tail(1)
    train = df.drop(index=test.index)
    # 仅保留训练集仍有 >= min_train 条交互的用户，保证可训练且可评估
    counts = train["u"].value_counts()
    valid_users = set(counts[counts >= min_train].index)
    train = train[train["u"].isin(valid_users)].reset_index(drop=True)
    test = test[test["u"].isin(valid_users)].reset_index(drop=True)
    return train, test


def build_train_sets(train: pd.DataFrame, n_users: int) -> list[set[int]]:
    """每个用户的训练正样本集合（用于候选集屏蔽）。"""
    sets: list[set[int]] = [set() for _ in range(n_users)]
    for u, i in zip(train["u"].to_numpy(), train["i"].to_numpy()):
        sets[u].add(i)
    return sets


def prepare_dataset(data_dir: Path, min_rating: float = 4.0) -> dict:
    """一站式数据准备：返回训练 / 评估所需的全部对象。"""
    data_path = download_ml100k(data_dir)
    df, u_uniq, i_uniq = load_interactions(data_path, min_rating)
    train, test = leave_one_out_split(df)
    n_users, n_items = len(u_uniq), len(i_uniq)

    train_u = train["u"].to_numpy()
    train_i = train["i"].to_numpy()
    # 稠密 seen 矩阵：负采样时做向量化屏蔽（数据量小，内存可承受）
    seen = np.zeros((n_users, n_items), dtype=bool)
    seen[train_u, train_i] = True

    return {
        "train_u": train_u,
        "train_i": train_i,
        "test_u": test["u"].to_numpy(),
        "test_i": test["i"].to_numpy(),
        "n_users": n_users,
        "n_items": n_items,
        "train_sets": build_train_sets(train, n_users),
        "seen": seen,
        "u_uniq": u_uniq,
        "i_uniq": i_uniq,
    }


def sample_negatives(
    users: np.ndarray,
    n_items: int,
    seen: np.ndarray,
    num_neg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """为一批用户各采样 num_neg 个不在其训练历史中的负样本 item（向量化拒绝采样）。"""
    neg = rng.integers(0, n_items, size=(len(users), num_neg))
    collision = seen[users[:, None], neg]
    # 反复重采样直到没有命中用户历史为止（收敛很快）
    while collision.any():
        neg[collision] = rng.integers(0, n_items, size=int(collision.sum()))
        collision = seen[users[:, None], neg]
    return neg
