"""
Transformer Text Classification — Pure PyTorch Implementation
=============================================================
A complete, self-contained Transformer encoder for binary sentiment
classification.  No Hugging Face dependencies — just torch and numpy.

Run:  python transformer_classifier.py
"""

from __future__ import annotations

import math
import random
from collections import Counter
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# 0.  Hyper-parameters  (centralised — tweak these to experiment)
# ---------------------------------------------------------------------------
D_MODEL: int = 128          # embedding / hidden dimension
NHEAD: int = 4              # number of attention heads
NUM_LAYERS: int = 2         # number of Transformer encoder layers
DIM_FF: int = 256           # feed-forward inner dimension
DROPOUT: float = 0.1        # dropout probability
MAX_SEQ_LEN: int = 32       # pad / truncate to this length
BATCH_SIZE: int = 16        # training & eval batch size
EPOCHS: int = 10            # maximum training epochs
LR: float = 1e-3            # peak learning rate (Adam)
WARMUP_STEPS: int = 50      # linear warmup steps
EARLY_STOP_PATIENCE: int = 3  # stop if no val improvement for N epochs
SEED: int = 42              # reproducibility seed

# ---------------------------------------------------------------------------
# 1.  Reproducibility helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ---------------------------------------------------------------------------
# 2.  Built-in toy dataset  (~100 English sentences, binary sentiment)
# ---------------------------------------------------------------------------
RAW_DATA: List[Tuple[str, int]] = [
    # ---- POSITIVE (label = 1) ----
    ("I love this movie", 1),
    ("this film is fantastic", 1),
    ("what a wonderful experience", 1),
    ("the acting was brilliant", 1),
    ("I really enjoyed the story", 1),
    ("great cinematography and music", 1),
    ("this is the best movie ever", 1),
    ("absolutely loved every moment", 1),
    ("the plot was engaging and fresh", 1),
    ("a beautiful romantic tale", 1),
    ("so heart warming and sweet", 1),
    ("the characters felt so real", 1),
    ("I was smiling throughout the film", 1),
    ("outstanding performances by all", 1),
    ("visually stunning and captivating", 1),
    ("makes you believe in true love", 1),
    ("hilarious from start to finish", 1),
    ("a masterpiece of modern cinema", 1),
    ("the dialogue was crisp and sharp", 1),
    ("I highly recommend this one", 1),
    ("it touched my heart deeply", 1),
    ("a powerful and moving drama", 1),
    ("the soundtrack was perfect", 1),
    ("every scene was a delight", 1),
    ("the director did a superb job", 1),
    ("a joy to watch again and again", 1),
    ("clever and unpredictable twists", 1),
    ("the animation was gorgeous", 1),
    ("I cried tears of happiness", 1),
    ("a must see for everyone", 1),
    ("the pacing was spot on", 1),
    ("wonderful storytelling all around", 1),
    ("exceeded all my expectations", 1),
    ("the chemistry between leads was electric", 1),
    ("fun for the whole family", 1),
    ("a delightful surprise", 1),
    ("one of the finest films this year", 1),
    ("truly inspiring and uplifting", 1),
    ("the special effects were mind blowing", 1),
    ("a perfect score from me", 1),
    ("had me hooked from the beginning", 1),
    ("a timeless classic already", 1),
    ("entertaining and meaningful", 1),
    ("richly layered and complex", 1),
    ("genuinely funny and smart", 1),
    ("an unforgettable cinematic journey", 1),
    ("the costumes and sets were lavish", 1),
    ("rivals the greats of the genre", 1),
    ("the ending was so satisfying", 1),
    ("restored my faith in filmmaking", 1),
    # ---- NEGATIVE (label = 0) ----
    ("I hate this movie", 0),
    ("this film is terrible", 0),
    ("what a boring experience", 0),
    ("the acting was awful", 0),
    ("I really disliked the story", 0),
    ("poor cinematography and music", 0),
    ("this is the worst movie ever", 0),
    ("absolutely hated every moment", 0),
    ("the plot was dull and predictable", 0),
    ("a terrible romantic cliché", 0),
    ("so disappointing and bland", 0),
    ("the characters felt fake", 0),
    ("I was bored throughout the film", 0),
    ("terrible performances by all", 0),
    ("ugly and unappealing visuals", 0),
    ("makes you cringe constantly", 0),
    ("not funny at all just lame", 0),
    ("a disaster of modern cinema", 0),
    ("the dialogue was cringeworthy", 0),
    ("I do not recommend this one", 0),
    ("it left me cold and empty", 0),
    ("a weak and forgettable drama", 0),
    ("the soundtrack was grating", 0),
    ("every scene was a drag", 0),
    ("the director clearly struggled", 0),
    ("painful to watch even once", 0),
    ("clumsy and obvious twists", 0),
    ("the animation was cheap looking", 0),
    ("I cried tears of frustration", 0),
    ("a skip for everyone", 0),
    ("the pacing was painfully slow", 0),
    ("awful storytelling all around", 0),
    ("far below my expectations", 0),
    ("the chemistry between leads was flat", 0),
    ("not fun for anyone", 0),
    ("a dreadful disappointment", 0),
    ("one of the worst films this year", 0),
    ("truly demoralizing and dull", 0),
    ("the special effects were laughable", 0),
    ("a zero from me", 0),
    ("losing interest from the beginning", 0),
    ("a forgettable mess already", 0),
    ("neither entertaining nor meaningful", 0),
    ("shallow and confusing", 0),
    ("genuinely unfunny and dumb", 0),
    ("a completely forgettable experience", 0),
    ("the costumes and sets were cheap", 0),
    ("nowhere near the greats of the genre", 0),
    ("the ending was infuriating", 0),
    ("destroyed my faith in filmmaking", 0),
]

# ---------------------------------------------------------------------------
# 3.  Simple tokeniser (lowercase → split on spaces / punctuation)
# ---------------------------------------------------------------------------
def simple_tokenize(text: str) -> List[str]:
    """Whitespace-and-punctuation split; decent for this tiny vocabulary."""
    text = text.lower().strip()
    # insert spaces around punctuation so they become separate tokens
    for ch in ".,!?;:'\"()[]{}":
        text = text.replace(ch, f" {ch} ")
    return text.split()


# Build vocabulary from the entire dataset
def build_vocab(
    sentences: List[str], min_freq: int = 1, max_size: int = 2000
) -> dict:
    """Create word → id mapping.  Special tokens: <PAD>=0, <UNK>=1."""
    counter: Counter = Counter()
    for sent in sentences:
        counter.update(simple_tokenize(sent))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, count) in enumerate(counter.most_common(max_size), start=2):
        if count < min_freq:
            break
        vocab[word] = i
    return vocab


# ---------------------------------------------------------------------------
# 4.  Dataset
# ---------------------------------------------------------------------------
class SentimentDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], vocab: dict, max_len: int):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        text, label = self.samples[idx]
        tokens = simple_tokenize(text)
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        # truncate
        ids = ids[: self.max_len]
        # pad
        pad_len = self.max_len - len(ids)
        ids = ids + [self.vocab["<PAD>"]] * pad_len
        mask = [1] * len(tokens[: self.max_len]) + [0] * pad_len
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# 5.  Positional encoding  (sinusoidal, fixed)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)          # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices
        pe = pe.unsqueeze(0)                            # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 6.  Transformer building blocks
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Vanilla multi-head scaled dot-product self-attention."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq, d_model) → (B, nhead, seq, d_k)
        B, S, _ = x.shape
        x = x.view(B, S, self.nhead, self.d_k).transpose(1, 2)
        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, nhead, seq, d_k) → (B, seq, d_model)
        B, _, S, _ = x.shape
        x = x.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return x

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, S, _ = x.shape
        Q = self._split_heads(self.W_q(x))  # (B, nhead, S, d_k)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, nhead, S, S)

        if mask is not None:
            # mask: (B, S)  →  (B, 1, 1, S) for broadcasting
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (B, nhead, S, d_k)
        context = self._combine_heads(context)     # (B, S, d_model)
        return self.W_o(context)


class FeedForwardNetwork(nn.Module):
    """Position-wise FFN:  Linear → ReLU → Linear → Dropout."""

    def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """One encoder layer:  Self-Attn + FFN, each with residual & LayerNorm."""

    def __init__(
        self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.ffn = FeedForwardNetwork(d_model, dim_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # --- Self-Attention sub-layer ---
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))   # residual + norm
        # --- Feed-Forward sub-layer ---
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))    # residual + norm
        return x


class TransformerEncoder(nn.Module):
    """Stack of `num_layers` encoder layers."""

    def __init__(
        self, num_layers: int, d_model: int, nhead: int, dim_ff: int, dropout: float
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ---------------------------------------------------------------------------
# 7.  Full classification model
# ---------------------------------------------------------------------------
class TransformerClassifier(nn.Module):
    """Embedding → PositionalEncoding → Encoder → mean-pool → Linear → logits."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dim_ff: int = DIM_FF,
        dropout: float = DROPOUT,
        max_seq_len: int = MAX_SEQ_LEN,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.encoder = TransformerEncoder(
            num_layers, d_model, nhead, dim_ff, dropout
        )
        self.classifier = nn.Linear(d_model, 2)  # binary

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.encoder(x, mask)
        # Mean-pool over the sequence (excluding padding positions)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # (B, S, 1)
            x = x * mask_expanded
            pooled = x.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return self.classifier(pooled)  # (B, 2)


# ---------------------------------------------------------------------------
# 8.  Learning-rate scheduler  (warmup + linear decay)
# ---------------------------------------------------------------------------
class WarmupLinearScheduler:
    """Linear warmup from 0 → peak_lr over warmup_steps, then linear decay to 0."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float = LR,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self) -> float:
        if self.current_step <= self.warmup_steps:
            return self.peak_lr * self.current_step / max(1, self.warmup_steps)
        progress = (self.current_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return self.peak_lr * max(0.0, 1.0 - progress)

    def get_last_lr(self) -> float:
        return self._get_lr()


# ---------------------------------------------------------------------------
# 9.  Training & evaluation helpers
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return accuracy on a given DataLoader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for input_ids, mask, labels in loader:
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, mask)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: WarmupLinearScheduler,
    device: torch.device,
) -> float:
    """Run one training epoch; return average loss."""
    model.train()
    total_loss = 0.0
    for input_ids, mask, labels in loader:
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
) -> nn.Module:
    """Full training loop with early-stopping and model checkpointing."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total_steps = epochs * len(train_loader)
    scheduler = WarmupLinearScheduler(optimizer, WARMUP_STEPS, total_steps, peak_lr=LR)

    best_val_acc = 0.0
    best_model_state = deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device
        )
        val_acc = evaluate(model, val_loader, device)

        print(
            f"  Epoch {epoch:3d}/{epochs}  |  "
            f"train loss: {train_loss:.4f}  |  "
            f"val acc: {val_acc:.4f}  |  lr: {scheduler.get_last_lr():.2e}"
        )

        # --- Early stopping logic ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping triggered after {epoch} epochs.")
                break

    # Restore best weights
    model.load_state_dict(best_model_state)
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    return model


# ---------------------------------------------------------------------------
# 10.  Inference helper
# ---------------------------------------------------------------------------
class Predictor:
    """Wraps the model and vocabulary for single-sentence inference."""

    def __init__(self, model: nn.Module, vocab: dict, device: torch.device):
        self.model = model
        self.vocab = vocab
        self.device = device

    def predict(self, text: str) -> int:
        """Return predicted class (0 or 1) for a single input sentence."""
        self.model.eval()
        tokens = simple_tokenize(text)
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        ids = ids[:MAX_SEQ_LEN]
        pad_len = MAX_SEQ_LEN - len(ids)
        ids = ids + [self.vocab["<PAD>"]] * pad_len
        mask = [1] * len(tokens[:MAX_SEQ_LEN]) + [0] * pad_len

        input_ids = torch.tensor([ids], dtype=torch.long).to(self.device)
        attn_mask = torch.tensor([mask], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attn_mask)
            pred = logits.argmax(dim=1).item()
        return pred


# ---------------------------------------------------------------------------
# 11.  Main — put it all together
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Total samples: {len(RAW_DATA)}")

    # ---- Split data ---- #
    indices = list(range(len(RAW_DATA)))
    random.shuffle(indices)
    n_train = int(0.8 * len(indices))
    n_val = int(0.1 * len(indices))

    train_samples = [RAW_DATA[i] for i in indices[:n_train]]
    val_samples = [RAW_DATA[i] for i in indices[n_train : n_train + n_val]]
    test_samples = [RAW_DATA[i] for i in indices[n_train + n_val :]]

    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # ---- Build vocabulary ---- #
    all_texts = [s for s, _ in RAW_DATA]
    vocab = build_vocab(all_texts)
    print(f"Vocabulary size: {len(vocab)}")

    # ---- DataLoaders ---- #
    train_ds = SentimentDataset(train_samples, vocab, MAX_SEQ_LEN)
    val_ds = SentimentDataset(val_samples, vocab, MAX_SEQ_LEN)
    test_ds = SentimentDataset(test_samples, vocab, MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Model ---- #
    model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_ff=DIM_FF,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total | {trainable_params:,} trainable")

    # ---- Train ---- #
    print("\nStarting training …")
    model = train(model, train_loader, val_loader, epochs=EPOCHS, device=device)

    # ---- Evaluate on test set ---- #
    test_acc = evaluate(model, test_loader, device)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # ---- Single-sample predictions ---- #
    predictor = Predictor(model, vocab, device)
    print("\n--- Inference examples ---")
    for sample_text in [
        "I love this movie",
        "I hate this film",
        "absolutely amazing and beautiful",
        "terrible boring and awful",
        "the story was okay I guess",
        "a delightful and heartwarming experience",
    ]:
        pred = predictor.predict(sample_text)
        label_str = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"  [{label_str}]  \"{sample_text}\"")


if __name__ == "__main__":
    main()
