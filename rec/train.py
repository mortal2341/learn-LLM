"""训练：BPR pairwise 排序损失 + 均匀负采样。

BPR（Bayesian Personalized Ranking）让正样本得分高于负样本：
    loss = -log sigmoid(score_pos - score_neg) = softplus(-(score_pos - score_neg))
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from data import sample_negatives


def train_model(
    model,
    bundle: dict,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    num_neg: int = 4,
    weight_decay: float = 1e-5,
    seed: int = 42,
    verbose: bool = True,
):
    """在隐式反馈训练集上训练双塔召回模型，返回训练好的模型。"""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_u = bundle["train_u"]
    train_i = bundle["train_i"]
    n_items = bundle["n_items"]
    seen = bundle["seen"]
    n_pairs = len(train_u)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_pairs)
        total_loss, n_batch = 0.0, 0

        for s in range(0, n_pairs, batch_size):
            idx = perm[s : s + batch_size]
            u_np, pos_np = train_u[idx], train_i[idx]
            neg_np = sample_negatives(u_np, n_items, seen, num_neg, rng)  # [B, num_neg]

            u = torch.from_numpy(u_np).to(device)
            pos = torch.from_numpy(pos_np).to(device)
            neg = torch.from_numpy(neg_np).to(device)

            pos_score = model(u, pos)  # [B]
            b = u.size(0)
            u_exp = u.unsqueeze(1).expand(-1, num_neg).reshape(-1)
            neg_score = model(u_exp, neg.reshape(-1)).reshape(b, num_neg)  # [B, num_neg]

            loss = F.softplus(-(pos_score.unsqueeze(1) - neg_score)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batch += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}/{epochs}  BPR loss = {total_loss / max(n_batch, 1):.4f}")

    return model
