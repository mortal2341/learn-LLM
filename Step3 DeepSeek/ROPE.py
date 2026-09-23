import torch
import torch.nn as nn


class ROPE(nn.Module):
    def __init__(self, dim, max_pos=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_pos = max_pos
        self.base = base
        inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 较小索引位置对应较低频率, 较大的索引位置有较高的频率
        self._set_cos_sin_cache(
            seq_len=max_pos,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len

        t = torch.arange(
            self.max_seq_len_cached, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # 注册到pytorch
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # 如果输入长度超过原定义的长度，则重新生成cos cache, sin cache，x: [bs, n_heads, seq_len, head_size]
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            # print("seq_len:", seq_len, self.max_seq_len_cached)
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """对输入的一半维度做旋转"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin, position_ids, unsqueeze_dim=1):
    """对q, k分别做位置旋转，即乘上旋转矩阵"""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)

    return q_embed