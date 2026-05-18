"""
model.py — Transformer Architecture Implementation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention.
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q    : (..., seq_q, d_k)
        K    : (..., seq_k, d_k)
        V    : (..., seq_k, d_v)
        mask : BoolTensor broadcastable to (..., seq_q, seq_k).
               True positions are masked out (set to -inf before softmax).
    Returns:
        output  : (..., seq_q, d_v)
        attn_w  : (..., seq_q, seq_k)  attention weights
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = F.softmax(scores, dim=-1)
    # Replace NaN that arise when an entire row is -inf (fully-masked)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Source padding mask.
    Returns shape (batch, 1, 1, src_len) — True where token == pad_idx.
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Target mask = padding mask OR causal (look-ahead) mask.
    Returns shape (batch, 1, tgt_len, tgt_len) — True where attention
    should be blocked.
    """
    tgt_len = tgt.size(1)
    device  = tgt.device
    # (B, 1, 1, T) — pad positions
    pad_mask    = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    # (1, 1, T, T) — upper-triangular causal mask
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need".
    Does NOT use torch.nn.MultiheadAttention internally.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, d_model) → (B, h, S, d_k)"""
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, h, S, d_k) → (B, S, d_model)"""
        B, _, S, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self._merge_heads(attn_out))

    def forward_with_weights(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same as forward() but also returns attention weights (B, h, S, S)."""
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))
        attn_out, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self._merge_heads(attn_out)), attn_w


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in "Attention Is All You Need".

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Registered as a buffer (not a trainable parameter).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (1, max_len, d_model) — registered as non-trainable buffer
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  LEARNED POSITIONAL ENCODING  (for Part-2 ablation 2.4)
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings via nn.Embedding.
    Drop-in replacement for PositionalEncoding (ablation 2.4).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.embedding(positions)
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Point-wise Feed-Forward Network.
    FFN(x) = max(0, x W1 + b1) W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  (Pre-LN: norm applied before sub-layer)
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.
    Uses Pre-LayerNorm (norm before sub-layer) for training stability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Pre-LN self-attention
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask))
        # Pre-LN feed-forward
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER  (Pre-LN)
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.
    Uses Pre-LayerNorm for training stability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-LN masked self-attention
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        # Pre-LN cross-attention
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, src_mask))
        # Pre-LN feed-forward
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

def _infer_N_from_state_dict(sd: dict) -> int:
    """Count encoder layers in a state_dict by finding max layer index."""
    max_idx = -1
    for k in sd.keys():
        if k.startswith("encoder.layers."):
            idx = int(k.split(".")[2])
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 3


def _infer_num_heads_from_state_dict(sd: dict, d_model: int) -> int:
    """Infer num_heads from W_q weight shape. W_q: (d_model, d_model) → heads = d_model / d_k."""
    # We can't directly get d_k from state_dict alone without more info,
    # so fall back to a safe default of 8 (or read from config in checkpoint).
    return 8


def _infer_d_ff_from_state_dict(sd: dict) -> int:
    """Infer d_ff from the first FFN linear1 weight shape."""
    for k, v in sd.items():
        if "ffn.linear1.weight" in k:
            return v.shape[0]
    return 512


class Transformer(nn.Module):
    """
    Full Transformer for sequence-to-sequence translation.

    When called with no arguments (Transformer()), downloads the saved
    checkpoint from Google Drive and reconstructs the exact architecture
    that was trained — including the correct number of layers (N).
    """

    # ── Replace with your actual Google Drive file ID after training ──
    _GDRIVE_FILE_ID  = "1lFE1VxzxMWaXlr8kiHHKJD_JnTdgP0RX"
    _CHECKPOINT_NAME = "baseline_best.pt"

    def __init__(
        self,
        src_vocab_size:   int   = None,
        tgt_vocab_size:   int   = None,
        d_model:          int   = 256,
        N:                int   = 4,       # default matches training config
        num_heads:        int   = 8,
        d_ff:             int   = 512,
        dropout:          float = 0.1,
        learned_pos_enc:  bool  = False,   # Part-2 ablation 2.4
        checkpoint_path:  str   = None,    # None = autograder path (download)
    ) -> None:
        super().__init__()

        # ── Inference / autograder path: Transformer() with no args ──
        if checkpoint_path is None and src_vocab_size is None:
            ckpt_path = self._CHECKPOINT_NAME
            if not os.path.exists(ckpt_path):
                gdown.download(id=self._GDRIVE_FILE_ID, output=ckpt_path, quiet=False)
            state = torch.load(ckpt_path, map_location="cpu")
            sd = state["model_state_dict"] if "model_state_dict" in state else state

            # ── Read all architecture dims from the checkpoint itself ──
            src_vocab_size = sd["src_embed.weight"].shape[0]
            tgt_vocab_size = sd["tgt_embed.weight"].shape[0]
            d_model        = sd["src_embed.weight"].shape[1]
            N              = _infer_N_from_state_dict(sd)
            d_ff           = _infer_d_ff_from_state_dict(sd)

            # num_heads: read from saved config if available, else infer
            if "model_config" in state and "num_heads" in state["model_config"]:
                num_heads = state["model_config"]["num_heads"]
            else:
                num_heads = _infer_num_heads_from_state_dict(sd, d_model)

            # learned_pos_enc: read from saved config if available
            if "model_config" in state:
                learned_pos_enc = state["model_config"].get("learned_pos_enc", False)

        else:
            sd = None  # training mode — no checkpoint to load yet

        # ── Store hyper-parameters ────────────────────────────────────
        self.d_model         = d_model
        self.src_vocab_size  = src_vocab_size
        self.tgt_vocab_size  = tgt_vocab_size
        self.N               = N
        self.num_heads       = num_heads
        self.d_ff            = d_ff
        self.dropout_rate    = dropout
        self.learned_pos_enc = learned_pos_enc

        # ── Embeddings ────────────────────────────────────────────────
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # ── Positional Encoding (sinusoidal or learned) ───────────────
        PE = LearnedPositionalEncoding if learned_pos_enc else PositionalEncoding
        self.src_pe = PE(d_model, dropout)
        self.tgt_pe = PE(d_model, dropout)

        # ── Encoder & Decoder stacks ──────────────────────────────────
        enc_layer    = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer    = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # ── Output projection ─────────────────────────────────────────
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

        if sd is not None:
            self.load_state_dict(sd)

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all weight matrices."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Core encode / decode ──────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ── Vocab loading helper for infer() ─────────────────────────────

    def _load_vocabs(self) -> None:
        if hasattr(self, '_vocabs_loaded'):
            return
        ckpt_path = self._CHECKPOINT_NAME
        if not os.path.exists(ckpt_path):
            gdown.download(id=self._GDRIVE_FILE_ID, output=ckpt_path, quiet=False)
        state = torch.load(ckpt_path, map_location="cpu")
        self._src_stoi = state["src_vocab"]
        self._tgt_itos = {i: t for t, i in state["tgt_vocab"].items()}
        self._vocabs_loaded = True

    # ── Autograder inference entry point ─────────────────────────────

    def infer(self, src_sentence: str) -> str:
        """
        Translate a German sentence to English using greedy decoding.

        Args:
            src_sentence : Raw German string.
        Returns:
            Translated English string.
        """
        self.eval()
        self._load_vocabs()
        device = next(self.parameters()).device

        unk_idx, pad_idx, sos_idx, eos_idx = 0, 1, 2, 3

        # Whitespace tokenisation (no spaCy dependency at inference)
        tokens  = src_sentence.lower().split()
        src_ids = [sos_idx] + [self._src_stoi.get(t, unk_idx) for t in tokens] + [eos_idx]

        src      = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = make_src_mask(src, pad_idx=pad_idx)

        with torch.no_grad():
            memory = self.encode(src, src_mask)

        tgt_indices = [sos_idx]
        max_len     = src.size(1) + 50

        with torch.no_grad():
            for _ in range(max_len):
                tgt      = torch.tensor([tgt_indices], dtype=torch.long, device=device)
                tgt_mask = make_tgt_mask(tgt, pad_idx=pad_idx)
                logits   = self.decode(memory, src_mask, tgt, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1).item()
                tgt_indices.append(next_tok)
                if next_tok == eos_idx:
                    break

        out = tgt_indices[1:]
        if eos_idx in out:
            out = out[:out.index(eos_idx)]
        return " ".join(self._tgt_itos.get(i, "<unk>") for i in out)