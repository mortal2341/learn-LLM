from dataclasses import dataclass
import torch
import torch.nn as nn
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

@dataclass
class AttentionConfig:
    d_model: int
    num_heads: int
    dropout: float = 0.1
    max_seq_len: int = 512


class MultiHeadAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.dropout_prob = config.dropout
        self.head_dim = self.d_model // self.num_heads

        assert self.head_dim * self.num_heads == self.d_model, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)

        # Project queries, keys, values
        Q = self.q_proj(query)  # (B, L_q, d_model)
        K = self.k_proj(key)  # (B, L_k, d_model)
        V = self.v_proj(value)  # (B, L_v, d_model)

        # Reshape and transpose for multi-head attention
        # (B, L, d_model) -> (B, L, num_heads, head_dim) -> (B, num_heads, L, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # (B, num_heads, L_q, head_dim) @ (B, num_heads, head_dim, L_k) = (B, num_heads, L_q, L_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores += attn_mask.unsqueeze(0).unsqueeze(1)  # Add mask to attention scores

        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, L_q, L_k)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        output = torch.matmul(attn_weights, V)  # (B, num_heads, L_q, head_dim)

        # Concatenate heads and apply final projection
        # (B, num_heads, L_q, head_dim) -> (B, L_q, num_heads, head_dim) -> (B, L_q, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)

        return output, attn_weights


def visualize_attention_weights(attn_weights, title="Attention Weights"):
    """
    Visualize attention weights using seaborn heatmap
    attn_weights: (num_heads, seq_len, seq_len) tensor
    """
    # Move to CPU and convert to numpy
    attn_weights_cpu = attn_weights.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)

    # Plot each attention head
    for i in range(min(4, attn_weights_cpu.shape[0])):  # Only plot up to 4 heads
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        sns.heatmap(
            attn_weights_cpu[i],
            ax=ax,
            cmap='viridis',
            cbar=True,
            xticklabels=[f'Token {j}' for j in range(attn_weights_cpu.shape[1])],
            yticklabels=[f'Token {j}' for j in range(attn_weights_cpu.shape[2])]
        )
        ax.set_title(f'Head {i + 1}')
        ax.tick_params(axis='x', rotation=90)

    # Hide unused subplots if there are fewer than 4 heads
    for i in range(attn_weights_cpu.shape[0], 4):
        row = i // 2
        col = i % 2
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()


def test_case():
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create synthetic dataset
    batch_size = 2
    seq_len = 8
    d_model = 64

    query = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
    value = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

    # Causal mask - upper triangular part set to -inf
    attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

    # Initialize MultiHeadAttention
    config = AttentionConfig(d_model=d_model, num_heads=4)
    mha = MultiHeadAttention(config)

    # Forward pass
    output, attn_weights = mha(query, key, value, attn_mask=attn_mask)

    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item()}")
    print(f"Attention weights shape: {attn_weights.shape}")

    # Visualize attention weights for the first batch item
    first_batch_attn_weights = attn_weights[0]  # (num_heads, seq_len, seq_len)
    visualize_attention_weights(first_batch_attn_weights, title="Multi-Head Attention Visualization")

    # Verify output shape
    expected_shape = (batch_size, seq_len, d_model)
    if output.shape != expected_shape:
        raise AssertionError(f"Expected output shape {expected_shape}, but got {output.shape}")

    print("Test passed!")


if __name__ == "__main__":
    test_case()
