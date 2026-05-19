# DA6401 — Assignment 3: Implementing the Transformer for Machine Translation

> **"Attention Is All You Need"** — Vaswani et al., 2017  
> Full Transformer implementation for German → English translation on the Multi30k dataset.

## Weights & Biases Report

[View W&B Report](https://api.wandb.ai/links/govindharshavardhan-iit-madras-iit-madras/3wta203i)

```
https://api.wandb.ai/links/govindharshavardhan-iit-madras-iit-madras/3wta203i
```

---

## Project Structure

```text
assignment3/
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── model.py               # Core Transformer architecture
├── train.py               # Training loop, greedy decoding, BLEU evaluation
├── dataset.py             # Multi30k dataset loading & spaCy tokenization
└── lr_scheduler.py        # Noam and Fixed LR schedulers
```

---

## Model Architecture (`model.py`)

A faithful re-implementation of the original Transformer paper.

| Component | Details |
|---|---|
| Scaled Dot-Product Attention | `Q·Kᵀ / √dₖ` with optional scaling (ablation 2.2) |
| Multi-Head Attention | Splits into `num_heads` heads, projects with W_q, W_k, W_v, W_o |
| Positional Encoding | Sinusoidal (default) or Learned Embedding (ablation 2.4) |
| Encoder Layer | Pre-norm → Self-Attention → FFN with residual connections |
| Decoder Layer | Pre-norm → Masked Self-Attn → Cross-Attn → FFN |
| Encoder / Decoder Stacks | N identical layers with final LayerNorm |
| Output Projection | Linear layer mapping `d_model → tgt_vocab_size` |

**Default hyperparameters:**

| Hyperparameter | Value |
|---|---|
| `d_model` | 256 |
| `N` (layers) | 3 |
| `num_heads` | 8 |
| `d_ff` | 512 |
| `dropout` | 0.1 |

**Autograder contract (do not modify these signatures):**
- `scaled_dot_product_attention(Q, K, V, mask)` → `(output, weights)`
- `MultiHeadAttention.forward(q, k, v, mask)` → `Tensor`
- `PositionalEncoding.forward(x)` → `Tensor`
- `make_src_mask(src, pad_idx)` → `BoolTensor`
- `make_tgt_mask(tgt, pad_idx)` → `BoolTensor`
- `Transformer.encode(src, src_mask)` → `Tensor`
- `Transformer.decode(memory, src_mask, tgt, tgt_mask)` → `Tensor`

---

## Dataset (`dataset.py`)

Uses the [Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) German→English dataset loaded via HuggingFace `datasets`.

- **Tokenization:** spaCy (`de_core_news_sm` for German, `en_core_web_sm` for English)
- **Vocabulary:** Built from training split only (no data leakage). Special tokens: `<unk>=0`, `<pad>=1`, `<sos>=2`, `<eos>=3`
- **DataLoaders:** `train_ds.get_dataloaders()` returns `(train_loader, val_loader, test_loader)` with padding collation

---

## Training (`train.py`)

### Loss Function
`LabelSmoothingLoss` — smoothed target distribution as per §5.4 of the paper.  
Set `--smoothing 0.0` for standard cross-entropy (ablation 2.5).

### Metrics Tracked (W&B)
- `train_loss` / `val_loss` per epoch
- `train_accuracy` / `val_accuracy` per epoch
- `train_pred_confidence` / `val_pred_confidence` — mean softmax prob of correct token (ablation 2.5)
- `val_bleu` — corpus-level BLEU score via greedy decoding (every `--val_bleu_freq` epochs)
- `test_bleu` — final test set BLEU
- `lr` — learning rate per step
- `grad_norm/W_q_mean`, `grad_norm/W_k_mean` — Q/K gradient norms (ablation 2.2, when `--log_grad_norm`)
- Attention heatmaps from last encoder layer logged every 5 epochs (ablation 2.3)

---

## Learning Rate Scheduler (`lr_scheduler.py`)

### Noam Scheduler (default)
From "Attention Is All You Need" §5.3:

```
lrate = d_model^(−0.5) × min(step^(−0.5), step × warmup_steps^(−1.5))
```

Linear warm-up for `warmup_steps` steps, then inverse-square-root decay.

### Fixed LR Scheduler
No-op scheduler — keeps LR constant. Used in ablation 2.1.

---

## Ablation Studies (Part 2)

| Flag | Options | Ablation |
|---|---|---|
| `--scheduler` | `noam` (default) / `fixed` | 2.1 — Noam vs Fixed LR |
| `--use_scale` | `true` (default) / `false` | 2.2 — Effect of √dₖ scaling |
| `--log_grad_norm` | flag | 2.2 — Log Q/K gradient norms |
| `--pos_encoding` | `sinusoidal` (default) / `learned` | 2.4 — Positional encoding type |
| `--smoothing` | `0.1` (default) / `0.0` | 2.5 — Label smoothing vs standard CE |

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---

## Training

```bash
# Default run (Noam scheduler, sinusoidal PE, label smoothing=0.1)
python train.py

# Ablation 2.1 — Fixed LR
python train.py --scheduler fixed --fixed_lr 1e-4

# Ablation 2.2 — Without sqrt(dk) scaling
python train.py --use_scale false --log_grad_norm

# Ablation 2.4 — Learned positional encoding
python train.py --pos_encoding learned

# Ablation 2.5 — No label smoothing
python train.py --smoothing 0.0
```

### Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--d_model` | 256 | Model dimensionality |
| `--N` | 3 | Number of encoder/decoder layers |
| `--num_heads` | 8 | Attention heads |
| `--d_ff` | 512 | Feed-forward hidden size |
| `--dropout` | 0.1 | Dropout rate |
| `--batch_size` | 128 | Training batch size |
| `--num_epochs` | 20 | Training epochs |
| `--warmup_steps` | 4000 | Noam warm-up steps |
| `--max_len` | 100 | Max decoding length |
| `--ckpt_dir` | `checkpoints/` | Checkpoint save directory |
| `--run_name` | auto-generated | W&B run name |

---

## Inference

```python
from model import Transformer

model = Transformer()   # auto-downloads best checkpoint from GDrive
translation = model.infer("Ein Mann sitzt auf einer Bank .")
print(translation)
```

---

## Requirements

```
torch, numpy, matplotlib, scikit-learn, wandb, datasets, spacy, tqdm, gdown, torchtext
```