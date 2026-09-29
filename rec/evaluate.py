"""召回生成与评估：全库打分 -> top-K 候选 -> Recall@K / HitRate@K / NDCG@K。

物品数较小（千级）时直接暴力全库打分 + torch.topk，无需引入 FAISS 等 ANN 库。
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def generate_candidates(model, bundle: dict, device: torch.device, topk: int = 50):
    """对每个用户在全库物品上打分，屏蔽训练集已交互物品，返回 top-K 候选。

    Returns:
        topk_items:  [n_users, topk] 候选物品的内部 id
        topk_scores: [n_users, topk] 对应匹配分
    """
    model.eval()
    n_users = bundle["n_users"]
    train_sets = bundle["train_sets"]

    item_mat = model.full_item_matrix(device)          # [n_items, dim]
    bias = model.item_bias.weight.squeeze(-1)          # [n_items]
    all_users = torch.arange(n_users, device=device)
    user_mat = model.user_vector(all_users)            # [n_users, dim]

    scores = user_mat @ item_mat.t() + bias            # [n_users, n_items]
    for u in range(n_users):
        seen_items = train_sets[u]
        if seen_items:
            scores[u, list(seen_items)] = -1e9  # 不推荐用户已交互过的物品

    topk_scores, topk_items = torch.topk(scores, k=topk, dim=1)
    return topk_items.cpu().numpy(), topk_scores.cpu().numpy()


def evaluate_recall(topk_items: np.ndarray, test_u: np.ndarray, test_i: np.ndarray, ks=(10, 20, 50)) -> dict:
    """留一法下每个用户只有 1 个相关物品，故 Recall@K 数值上等于 HitRate@K。"""
    results = {}
    n = len(test_u)
    for k in ks:
        hits, ndcg = 0, 0.0
        for u, it in zip(test_u, test_i):
            row = topk_items[u, :k]
            match = np.where(row == it)[0]
            if match.size > 0:
                hits += 1
                rank = int(match[0]) + 1
                ndcg += 1.0 / np.log2(rank + 1)
        results[k] = {"recall": hits / n, "hitrate": hits / n, "ndcg": ndcg / n}
    return results
