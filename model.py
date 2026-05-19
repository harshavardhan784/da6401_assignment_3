# model.py

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

CHANGES vs submitted version:
  1. scaled_dot_product_attention: added `use_scale` flag (default True)
     → ablation 2.2 (with/without √dk scaling factor).
  2. MultiHeadAttention: now stores last attention weights in
     self.attn_weights so they can be extracted for ablation 2.3
     (attention heatmaps) and 2.2 (gradient norm logging).
  3. Added LearnedPositionalEncoding (nn.Embedding-based) for ablation 2.4.
  4. Transformer.__init__: added `pos_encoding` arg ('sinusoidal'|'learned')
     and `use_scale` arg; both forwarded through the stack.
  5. Encoder/Decoder stacks: pass use_scale down to every MHA layer.
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
#  SCALED DOT-PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scale: bool = True,                    # FIX: ablation 2.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled (or unscaled) dot-product attention.

    Args:
        Q, K, V    : query / key / value tensors  [B, heads, S, d_k]
        mask       : boolean mask; True positions are filled with -inf
        use_scale  : if False, skip the 1/√d_k scaling (ablation 2.2)

    Returns:
        output      : [B, heads, S, d_k]
        attn_weights: [B, heads, S_q, S_k]  — needed for 2.3 heatmaps
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scale:                              # FIX: ablation 2.2
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    # (batch, src_len) → (batch, 1, 1, src_len)
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    tgt_len = tgt.size(1)
    device  = tgt.device
    pad_mask    = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)          # (B,1,1,T)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)                                        # (1,1,T,T)
    return pad_mask | causal_mask                                      # (B,1,T,T)


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        dropout:   float = 0.1,
        use_scale: bool  = True,               # FIX: ablation 2.2
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.use_scale = use_scale             # FIX: ablation 2.2

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # FIX: store last attention weights for heatmap extraction (ablation 2.3)
        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
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

        # FIX: pass use_scale; capture attn_weights for ablations 2.2/2.3
        attn_out, attn_w = scaled_dot_product_attention(
            Q, K, V, mask, use_scale=self.use_scale
        )
        self.attn_weights = attn_w.detach()    # FIX: store for external access

        return self.W_o(self._merge_heads(attn_out))


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODINGS  (sinusoidal + learned)
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — original paper, ablation 2.4 baseline."""

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
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding via nn.Embedding — ablation 2.4.

    Replaces the fixed sinusoidal PE with trainable position embeddings.
    Maximum sequence length is capped at max_len (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)  # trainable, NOT a buffer
        nn.init.normal_(self.pos_embed.weight, mean=0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len  = x.size(1)
        device   = x.device
        positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        x = x + self.pos_embed(positions)
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.1,
        use_scale: bool  = True,               # FIX: ablation 2.2
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm: normalise BEFORE the sub-layer (more stable for Transformers)
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.1,
        use_scale: bool  = True,               # FIX: ablation 2.2
    ) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
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
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, src_mask))
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

class Transformer(nn.Module):

    # UPDATE these after uploading your best checkpoint to GDrive
    # _GDRIVE_FILE_ID  = "1EHoboX_JUdHVC3HkLM4LTC-eMCBqvo1E"   # replace with new file id
    _GDRIVE_FILE_ID = "1QcGyNTPEzje7_66CXzhRSJzSQtKBnAzy"
    _CHECKPOINT_NAME = "checkpoint_epoch17.pt"                 # best val loss = 2.8796

    def __init__(
        self,
        src_vocab_size:  int   = None,
        tgt_vocab_size:  int   = None,
        d_model:         int   = 256,
        N:               int   = 3,
        num_heads:       int   = 8,
        d_ff:            int   = 512,
        dropout:         float = 0.1,
        pos_encoding:    str   = 'sinusoidal',   # FIX: ablation 2.4 ('sinusoidal'|'learned')
        use_scale:       bool  = True,           # FIX: ablation 2.2
        checkpoint_path: str   = None,
    ) -> None:
        super().__init__()

        # ── Inference mode: load everything from checkpoint ───────────
        if checkpoint_path is None and src_vocab_size is None:
            ckpt = self._CHECKPOINT_NAME
            if not os.path.exists(ckpt):
                gdown.download(id=self._GDRIVE_FILE_ID, output=ckpt, quiet=False)
            state = torch.load(ckpt, map_location="cpu")
            sd    = state["model_state_dict"] if "model_state_dict" in state else state

            # Read ALL architecture params from saved model_config so that
            # Transformer() with no args always matches the checkpoint exactly,
            # regardless of what defaults are written here.
            cfg = state.get("model_config", {})
            src_vocab_size = cfg.get("src_vocab_size", sd["src_embed.weight"].shape[0])
            tgt_vocab_size = cfg.get("tgt_vocab_size", sd["tgt_embed.weight"].shape[0])
            d_model        = cfg.get("d_model",        sd["src_embed.weight"].shape[1])
            N              = cfg.get("N",              N)
            num_heads      = cfg.get("num_heads",      num_heads)
            d_ff           = cfg.get("d_ff",           d_ff)
            dropout        = cfg.get("dropout",        dropout)
            pos_encoding   = cfg.get("pos_encoding",   pos_encoding)
            use_scale      = cfg.get("use_scale",      use_scale)
        else:
            sd = None

        self.d_model        = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.N              = N
        self.num_heads      = num_heads
        self.d_ff           = d_ff
        self.dropout_rate   = dropout
        self.pos_encoding   = pos_encoding      # FIX: store for save_checkpoint
        self.use_scale      = use_scale         # FIX: store for save_checkpoint

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # FIX: switch between sinusoidal and learned PE (ablation 2.4)
        def _make_pe():
            if pos_encoding == 'learned':
                return LearnedPositionalEncoding(d_model, dropout)
            return PositionalEncoding(d_model, dropout)

        self.src_pe = _make_pe()
        self.tgt_pe = _make_pe()

        # FIX: pass use_scale into every encoder/decoder layer (ablation 2.2)
        enc_layer    = EncoderLayer(d_model, num_heads, d_ff, dropout, use_scale)
        dec_layer    = DecoderLayer(d_model, num_heads, d_ff, dropout, use_scale)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

        if sd is not None:
            self.load_state_dict(sd)

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

    def _load_vocabs(self):
        if hasattr(self, '_vocabs_loaded'):
            return
        ckpt = self._CHECKPOINT_NAME
        if not os.path.exists(ckpt):
            gdown.download(id=self._GDRIVE_FILE_ID, output=ckpt, quiet=False)
        state = torch.load(ckpt, map_location="cpu")
        self._src_stoi = state["src_vocab"]
        self._tgt_itos = {i: t for t, i in state["tgt_vocab"].items()}
        self._vocabs_loaded = True

    def infer(self, src_sentence: str) -> str:
        self.eval()
        self._load_vocabs()
        device = next(self.parameters()).device

        unk_idx, pad_idx, sos_idx, eos_idx = 0, 1, 2, 3

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