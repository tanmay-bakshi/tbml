import argparse
import html
import json
import os
from datetime import datetime
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tbml.experiments.honeycomb.text.dataset import _build_tokenizer
from tbml.experiments.honeycomb.text.model import TextTransformer, TextTransformerConfig


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Visualize token embeddings with PCA for a LeJEPA text checkpoint."
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory or model.eqx file.")
    parser.add_argument("text", type=str, help="Input text string to visualize.")
    parser.add_argument("--output", type=str, default=None, help="Optional output HTML path.")
    return parser.parse_args()


def _dtype_from_name(name: str) -> jnp.dtype:
    """Map a dtype name to a JAX dtype.

    :param name: Dtype name string.
    :returns: JAX dtype.
    """
    if name == "float32":
        return jnp.float32
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float16":
        return jnp.float16
    raise ValueError(f"unsupported dtype: {name}")


def _resolve_checkpoint_dir(path: str) -> str:
    """Resolve a checkpoint directory from a user-provided path.

    :param path: Path to a checkpoint directory or a checkpoint file.
    :returns: Resolved checkpoint directory.
    """
    if os.path.isdir(path) is True:
        ckpt_dir = path
    else:
        ckpt_dir = os.path.dirname(path)
        if ckpt_dir == "":
            raise ValueError("checkpoint path must be a directory or a file inside a directory")

    model_path = os.path.join(ckpt_dir, "model.eqx")
    metadata_path = os.path.join(ckpt_dir, "metadata.json")
    if os.path.isfile(model_path) is False:
        raise FileNotFoundError(f"model.eqx not found in checkpoint directory: {ckpt_dir}")
    if os.path.isfile(metadata_path) is False:
        raise FileNotFoundError(f"metadata.json not found in checkpoint directory: {ckpt_dir}")
    return ckpt_dir


def _load_run_config(checkpoint_dir: str) -> dict[str, Any]:
    """Load the training run configuration from a checkpoint.

    :param checkpoint_dir: Checkpoint directory path.
    :returns: Run configuration dictionary.
    """
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    config = metadata.get("config")
    if config is None:
        raise ValueError("metadata.json does not contain a config entry")
    if isinstance(config, dict) is False:
        raise ValueError("metadata config entry must be a dictionary")
    return config


def _load_model(
    checkpoint_dir: str,
    *,
    dtype: jnp.dtype,
    model_config: dict[str, Any],
) -> TextTransformer:
    """Load a TextTransformer model from a checkpoint.

    :param checkpoint_dir: Checkpoint directory path.
    :param dtype: Compute and parameter dtype.
    :param model_config: Model configuration dictionary.
    :returns: Deserialized TextTransformer model.
    """
    config = TextTransformerConfig(**model_config)
    model_path = os.path.join(checkpoint_dir, "model.eqx")
    candidates = [dtype, jnp.float32, jnp.bfloat16, jnp.float16]
    seen: set[jnp.dtype] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        model_key = jax.random.PRNGKey(0)
        model = TextTransformer(
            config,
            dtype=candidate,
            param_dtype=candidate,
            key=model_key,
        )
        try:
            return eqx.tree_deserialise_leaves(model_path, model)
        except RuntimeError as exc:
            last_error = exc
            if "changed dtype" not in str(exc):
                raise
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("failed to load model checkpoint")


def _tokenize_text(
    text: str,
    *,
    tokenizer_name: str,
    eos_token: str,
    pad_token: str,
    mask_token: str,
    max_seq_len: int,
) -> tuple[list[int], list[str], int]:
    """Tokenize text the same way as the streaming dataset.

    :param text: Input text string.
    :param tokenizer_name: Hugging Face tokenizer identifier.
    :param eos_token: EOS token string.
    :param pad_token: Padding token string.
    :param mask_token: Masking token string.
    :param max_seq_len: Maximum sequence length.
    :returns: Tuple of (token ids, decoded token strings, pad token id).
    """
    tokenizer, eos_id, pad_id, _mask_id = _build_tokenizer(
        tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
    )
    encoding = tokenizer.encode(text, add_special_tokens=False)
    token_ids = list(encoding.ids)
    if len(token_ids) > max_seq_len - 1:
        token_ids = token_ids[: max_seq_len - 1]
    token_ids.append(eos_id)

    token_strings: list[str] = []
    for token_id in token_ids:
        decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        if len(decoded) == 0:
            decoded = eos_token
        token_strings.append(decoded)
    return token_ids, token_strings, pad_id


def _prepare_batch(
    token_ids: list[int],
    *,
    max_seq_len: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad token ids into a batch and build the attention mask.

    :param token_ids: Token ids for a single sample.
    :param max_seq_len: Maximum sequence length.
    :param pad_id: Padding token id.
    :returns: Tuple of (tokens, attention_mask) arrays.
    """
    if len(token_ids) > max_seq_len:
        raise ValueError("token_ids length exceeds max_seq_len")
    tokens = np.full((1, max_seq_len), pad_id, dtype=np.int32)
    attention_mask = np.zeros((1, max_seq_len), dtype=np.bool_)
    length = len(token_ids)
    if length > 0:
        tokens[0, :length] = np.asarray(token_ids, dtype=np.int32)
        attention_mask[0, :length] = True
    return tokens, attention_mask


def _compute_token_embeddings(
    model: TextTransformer,
    tokens: np.ndarray,
    attention_mask: np.ndarray,
    *,
    dtype: jnp.dtype,
) -> np.ndarray:
    """Compute token embeddings before pooling and final normalization.

    :param model: Text transformer model.
    :param tokens: Token ids array of shape (1, T).
    :param attention_mask: Attention mask array of shape (1, T).
    :param dtype: Compute dtype.
    :returns: Token embeddings of shape (T, d_model).
    """
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (1, T)")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape (1, T)")
    token_tensor = jnp.asarray(tokens, dtype=jnp.int32)
    mask_tensor = jnp.asarray(attention_mask, dtype=jnp.bool_)
    embeddings = model.encode_tokens(token_tensor, mask_tensor, train=False, key=None)
    embeddings = jax.device_get(embeddings)
    return np.asarray(embeddings, dtype=np.float32)[0]


def _pca_to_rgb(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 3D via PCA and scale to RGB.

    :param embeddings: Token embeddings of shape (num_tokens, dim).
    :returns: Array of shape (num_tokens, 3) scaled to [0, 255].
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (num_tokens, dim)")
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, v_t = np.linalg.svd(centered, full_matrices=False)
    comps = v_t[: min(3, v_t.shape[0]), :]
    scores = centered @ comps.T
    if scores.shape[1] < 3:
        pad = np.zeros((scores.shape[0], 3 - scores.shape[1]), dtype=scores.dtype)
        scores = np.concatenate([scores, pad], axis=1)

    colors = np.empty_like(scores, dtype=np.float32)
    for idx in range(3):
        channel = scores[:, idx]
        min_val = float(channel.min())
        max_val = float(channel.max())
        if max_val > min_val:
            scaled = (channel - min_val) / (max_val - min_val)
        else:
            scaled = np.full(channel.shape, 0.5, dtype=np.float32)
        colors[:, idx] = scaled

    colors = np.clip(colors * 255.0, 0.0, 255.0)
    return colors.astype(np.uint8)


def _color_text_html(tokens: list[str], colors: np.ndarray) -> str:
    """Render HTML with token background colors.

    :param tokens: Token text strings.
    :param colors: RGB values of shape (num_tokens, 3).
    :returns: HTML string for the token visualization.
    """
    if len(tokens) != colors.shape[0]:
        raise ValueError("tokens length must match colors shape")

    spans: list[str] = []
    for token_text, color in zip(tokens, colors, strict=True):
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "#000000" if luminance >= 128.0 else "#ffffff"
        escaped = html.escape(token_text)
        spans.append(
            "<span class=\"token\" "
            f"style=\"background-color: rgb({r}, {g}, {b}); color: {text_color};\">"
            f"{escaped}</span>"
        )

    body = "".join(spans)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Token PCA Visualization</title>
  <style>
    body {{
      font-family: "IBM Plex Mono", "SFMono-Regular", Menlo, Monaco, Consolas, monospace;
      margin: 24px;
      background: #f9f7f2;
      color: #222;
    }}
    .container {{
      white-space: pre-wrap;
      line-height: 1.5;
      font-size: 15px;
    }}
    .token {{
      padding: 0 2px;
      border-radius: 2px;
    }}
  </style>
</head>
<body>
  <div class="container">{body}</div>
</body>
</html>
"""


def _default_output_path() -> str:
    """Build a default output HTML path.

    :returns: Output HTML file path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.abspath(f"token_pca_{timestamp}.html")


def main() -> None:
    """Run the token PCA visualization."""
    args = _parse_args()
    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint)
    config = _load_run_config(checkpoint_dir)

    model_config = config.get("model")
    if model_config is None or isinstance(model_config, dict) is False:
        raise ValueError("run config missing model configuration")
    data_config = config.get("data")
    if data_config is None or isinstance(data_config, dict) is False:
        raise ValueError("run config missing data configuration")
    training_config = config.get("training")
    if training_config is None or isinstance(training_config, dict) is False:
        raise ValueError("run config missing training configuration")

    dtype_name = training_config.get("dtype")
    if dtype_name is None or isinstance(dtype_name, str) is False:
        raise ValueError("training config missing dtype string")
    dtype = _dtype_from_name(dtype_name)

    tokenizer_name = data_config.get("tokenizer")
    eos_token = data_config.get("eos_token")
    pad_token = data_config.get("pad_token")
    mask_token = data_config.get("mask_token")
    max_seq_len = data_config.get("max_seq_len")
    if tokenizer_name is None or isinstance(tokenizer_name, str) is False:
        raise ValueError("data config missing tokenizer name")
    if eos_token is None or isinstance(eos_token, str) is False:
        raise ValueError("data config missing eos token")
    if pad_token is None or isinstance(pad_token, str) is False:
        raise ValueError("data config missing pad token")
    if mask_token is None or isinstance(mask_token, str) is False:
        raise ValueError("data config missing mask token")
    if max_seq_len is None or isinstance(max_seq_len, int) is False:
        raise ValueError("data config missing max_seq_len")

    model = _load_model(checkpoint_dir, dtype=dtype, model_config=model_config)

    token_ids, token_texts, pad_id = _tokenize_text(
        args.text,
        tokenizer_name=tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
        max_seq_len=max_seq_len,
    )
    tokens, attention_mask = _prepare_batch(token_ids, max_seq_len=max_seq_len, pad_id=pad_id)
    embeddings = _compute_token_embeddings(model, tokens, attention_mask, dtype=dtype)
    embeddings = embeddings[: len(token_ids)]

    colors = _pca_to_rgb(embeddings)
    html_output = _color_text_html(token_texts, colors)

    output_path = args.output
    if output_path is None:
        output_path = _default_output_path()
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html_output)
    print(f"Wrote HTML visualization to: {output_path}")


if __name__ == "__main__":
    main()
