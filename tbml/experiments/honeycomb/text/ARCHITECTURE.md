## Base Model (train.py)

This section documents the text base model trained by `tbml/experiments/honeycomb/text/train.py`.
It summarizes the data format, preprocessing, model architecture, losses, and optimization as
implemented in the current code.

### Data format and batching

- Training data is pre-tokenized and stored as fixed-length NumPy shards (`shard_*.npy`) with
  shape `(num_samples, max_seq_len)`. The dataset is loaded via `MMapTokenDataset`, which memory
  maps shards and provides random access.
- Each sample is already padded to `max_seq_len` with a padding token. The dataset metadata
  includes `vocab_size`, `max_seq_len`, and the special token ids (`eos_id`, `pad_id`, `mask_id`).
- Batches are produced by `iter_text_batches`, which performs a streaming shuffle using a
  bounded buffer. The training script reshapes each host batch into per-device slices and
  prefetches to device.
- EOS tokens are stripped before the model forward pass by replacing EOS ids with the padding
  id. This means the model never attends to EOS positions during training.

### View generation and dropout

- TJepa-style training uses **global** and **local** views per sample.
  - One global view is always a **sample-level** view with no dropout.
  - Additional global views and all local views are identical token sequences, but are run
    through the model with different **input dropout** masks.
- Each view is assigned an input‑dropout rate sampled uniformly between a per‑view min/max:
  - `--global-dropout-min/max` for global views (excluding the sample view, which is always 0).
  - `--local-dropout-min/max` for local views.
- Input dropout is applied to the token embeddings before the transformer stack. Attention
  masks still exclude padding positions, but no tokens are hidden or removed.
- The predictor still supports learned position‑specific mask tokens, but under this
  architecture no positions are masked, so the predictor receives the encoder outputs directly.

### Model architecture

The base model is `TextTransformer` (see `tbml/experiments/honeycomb/text/model.py`). It is
composed of an **encoder** and an optional **predictor**.

- **Token embedding**
  - Embedding table of shape `(vocab_size, d_model)`.
  - Embedding normalization is configurable in the model config; current training sets
    `embed_norm=False` (no post-lookup RMSNorm).
  - The embedding module exposes an `unembed()` method that can project embeddings back to
    vocabulary logits using the raw embedding weights and a tied bias.
- **Transformer blocks**
  - Pre-norm RMSNorm.
  - Self-attention with PoPE or RoPE positional encoding (configured via `attn_type`).
  - Attention is configured independently for encoder and predictor via
    `--encoder-causal-attention` and `--predictor-causal-attention`.
  - Attention respects the provided attention masks (pads only, in this architecture).
  - MLP: SwiGLU with expansion ratio `mlp_ratio`.
  - DropPath is applied with a linear schedule over depth.
- **Encoder output head**
  - Encoder outputs follow: `[transformer] -> [final RMSNorm] -> [SwiGLU FFN]`.
  - The encoder’s token-level outputs used by training and inference are the **post‑head**
    representations (after the SwiGLU FFN).
- **Pooling**
  - The pooled sequence embedding is the representation of the **last valid position** according
    to the attention mask.
- **Predictor**
  - A separate transformer stack that operates on token‑level representations from the encoder
    output head (post‑head). Causality follows `--predictor-causal-attention`.
  - Number of predictor layers is configurable independently (`predictor_n_layers`); other
    hyperparameters (width, heads, MLP ratio, dropout, positional encoding) mirror the encoder.
  - If mask positions are provided, each masked position is replaced by a learned
    position‑specific embedding. With dropout‑only views, mask positions are empty.
  - Predictor outputs are also post‑head: `[predictor blocks] -> [predictor final RMSNorm] ->
    [predictor SwiGLU FFN]`.
- Predictor is optional: if `predictor_n_layers` is set to 0, it is not instantiated and any
  loss depending on it must be disabled.

The model returns both per-token representations and the pooled representation on every call.

### Losses

Let `B` be batch size, `V` be total number of views (global + local), and `K` the embedding
size. The encoder produces token representations of shape `(B, V, T, K)`.

1) **TJepa reconstruction loss (`tjepa_rec_loss`)**

- The encoder processes all global and local views together.
- The sample‑level global view is encoder‑only. Additional global views and all local views are
  passed through the predictor.
- A **token‑wise global center** is computed by averaging the **SWA encoder outputs** of all
  global views. The sample view is run through SWA with `train=False`, while the other global
  views use the same input‑dropout masks as the main model. The center is treated as a detached
  target (no gradient flow into SWA).
- Reconstruction matches **predictor outputs only** (non‑sample globals + locals) token‑by‑token
  to this global center. The sample‑level global view is excluded from reconstruction.
- A random subset of positions is included (controlled by `--tjepa-unmasked-keep-prob`). EOS and
  padding positions are excluded.

2) **TJepa SIGReg (`tjepa_sigreg_loss`)**

- For each view, a view‑level representation is computed by mean‑pooling the encoder’s token
  outputs over the attention mask (pads excluded).
- SIGReg is computed over the `(B, V, K)` tensor of view‑level representations.

3) **TJepa loss (`tjepa_loss`)**

```
tjepa_loss = (1 - sigreg_weight) * tjepa_rec_loss + sigreg_weight * tjepa_sigreg_loss
```

4) **Encoder MLM loss (`encoder_mlm_loss`)**

- The encoder’s post‑head token representation is unembedded through the tied token embedding
  weights and optimized to predict the original token id at selected positions.
- With dropout‑only views, no positions are masked by default; if `encoder_mlm_keep_prob > 0`,
  a random fraction of unmasked tokens contribute to the loss. Top‑1/top‑5 accuracy are
  reported for masked positions only, so they will be 0 when no mask positions exist.

5) **Total loss (`total_loss`)**

```
total_loss = tjepa_frac * tjepa_loss + mlm_frac * encoder_mlm_loss
```

The fractions are derived from the CLI flags `--tjepa-loss-weight` and
`--encoder-mlm-loss-weight` by normalizing them to sum to 1.0. When a weight is 0, that loss is
skipped entirely.

### Optimization and training loop

- Optimizer: `MuonWithAdamWFallback`.
  - Muon is applied to most parameters.
  - AdamW is used for excluded parameter groups (token embeddings, norms, and PoPE delta).
- Gradient accumulation is supported; micro-batches are stacked and reduced on device.
- Training uses `jax.pmap` across devices, with gradients and losses averaged across the
  `"data"` axis.
- The training loop performs a single full pass through the dataset (one epoch), unless
  `--max-train-steps` truncates it.
- Checkpoints are saved every `--checkpoint-every` steps and include model weights,
  optimizer state, and the run config.

### Precision

- Compute dtype is configurable (`float32`, `bfloat16`, `float16`).
- When using reduced precision, model parameters are stored in `float32`.
- Embeddings are cast to `float32` before loss computation to keep the loss path stable.
