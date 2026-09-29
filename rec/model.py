"""双塔召回模型：用户塔与物品塔分别把 id 映射为向量，打分 = 点积 + 物品偏置。

仅有 id 特征时，双塔在结构上退化为「带塔 MLP 的矩阵分解」，是召回的经典基线。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TwoTowerRecall(nn.Module):
    """Two-tower retrieval model.

    Args:
        n_users: 用户数。
        n_items: 物品数。
        embed_dim: 召回向量维度（用户与物品向量同维，便于点积）。
        hidden_dim: 塔内 MLP 隐层维度。
    """

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.n_items = n_items
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.item_bias = nn.Embedding(n_items, 1)

        self.user_tower = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def user_vector(self, user_ids: torch.Tensor) -> torch.Tensor:
        """用户 id -> 用户召回向量 [B, embed_dim]。"""
        return self.user_tower(self.user_emb(user_ids))

    def item_vector(self, item_ids: torch.Tensor) -> torch.Tensor:
        """物品 id -> 物品召回向量 [B, embed_dim]。"""
        return self.item_tower(self.item_emb(item_ids))

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """返回 (user, item) 匹配分 [B]。"""
        u = self.user_vector(user_ids)
        v = self.item_vector(item_ids)
        bias = self.item_bias(item_ids).squeeze(-1)
        return (u * v).sum(dim=-1) + bias

    def full_item_matrix(self, device: torch.device) -> torch.Tensor:
        """返回所有物品的塔向量 [n_items, embed_dim]，用于全库打分 / 召回。"""
        all_items = torch.arange(self.n_items, device=device)
        return self.item_vector(all_items)
