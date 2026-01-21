import argparse
import json
import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml

from tbml.experiments.honeycomb.text.dataset import _build_tokenizer
from tbml.experiments.honeycomb.text.model import TextTransformer, TextTransformerConfig


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Plot PCA of pooled sequence embeddings from a YAML list."
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory or model.eqx file.")
    parser.add_argument("yaml_path", type=str, help="Path to YAML file with label/content entries.")
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


def _load_yaml_entries(path: str) -> list[dict[str, str]]:
    """Load YAML entries from a file.

    :param path: Path to the YAML file.
    :returns: List of dicts with label/content.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if isinstance(data, list) is False:
        raise ValueError("YAML must contain a list of objects")
    entries: list[dict[str, str]] = []
    for idx, item in enumerate(data):
        if isinstance(item, dict) is False:
            raise ValueError(f"entry {idx} is not an object")
        label = item.get("label")
        content = item.get("content")
        if isinstance(label, str) is False or isinstance(content, str) is False:
            raise ValueError(f"entry {idx} must have string label and content")
        entries.append({"label": label, "content": content})
    if len(entries) == 0:
        raise ValueError("no entries found in YAML file")
    return entries


def _prepare_batch(
    entries: list[dict[str, str]],
    *,
    tokenizer_name: str,
    eos_token: str,
    pad_token: str,
    mask_token: str,
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[list[int]]]:
    """Tokenize and pad entries into a batch.

    :param entries: List of label/content dicts.
    :param tokenizer_name: Hugging Face tokenizer identifier.
    :param eos_token: EOS token string.
    :param pad_token: Padding token string.
    :param mask_token: Masking token string.
    :param max_seq_len: Maximum sequence length.
    :returns: Tuple of (tokens, attention_mask, labels, token_ids_per_sample).
    """
    tokenizer, eos_id, pad_id, _mask_id = _build_tokenizer(
        tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
    )

    batch_size = len(entries)
    tokens = np.full((batch_size, max_seq_len), pad_id, dtype=np.int32)
    attention_mask = np.zeros((batch_size, max_seq_len), dtype=np.bool_)
    labels: list[str] = []
    token_ids_per_sample: list[list[int]] = []
    for row, entry in enumerate(entries):
        labels.append(entry["label"])
        encoding = tokenizer.encode(entry["content"], add_special_tokens=False)
        token_ids = list(encoding.ids)
        if len(token_ids) > max_seq_len - 1:
            token_ids = token_ids[: max_seq_len - 1]
        token_ids_per_sample.append(token_ids)
        length = min(len(token_ids), max_seq_len)
        if length > 0:
            tokens[row, :length] = np.asarray(token_ids[:length], dtype=np.int32)
            attention_mask[row, :length] = True
    return tokens, attention_mask, labels, token_ids_per_sample


def _pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D via PCA.

    :param embeddings: Embeddings of shape (N, D).
    :returns: Array of shape (N, 2).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N, D)")
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, v_t = np.linalg.svd(centered, full_matrices=False)
    components = v_t[: min(2, v_t.shape[0]), :]
    scores = centered @ components.T
    if scores.shape[1] < 2:
        pad = np.zeros((scores.shape[0], 2 - scores.shape[1]), dtype=scores.dtype)
        scores = np.concatenate([scores, pad], axis=1)
    return scores


def _pca_to_rgb(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 3D via PCA and scale to RGB.

    :param embeddings: Embeddings of shape (N, D).
    :returns: Array of shape (N, 3) scaled to [0, 255].
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N, D)")
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, v_t = np.linalg.svd(centered, full_matrices=False)
    components = v_t[: min(3, v_t.shape[0]), :]
    scores = centered @ components.T
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


def _plot_embeddings(points: np.ndarray, labels: list[str]) -> None:
    """Render a labelled scatter plot.

    :param points: PCA points of shape (N, 2).
    :param labels: Label strings of length N.
    """
    if points.shape[0] != len(labels):
        raise ValueError("points and labels must have matching lengths")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(points[:, 0], points[:, 1], s=70, color="#277da1")
    for idx, label in enumerate(labels):
        ax.annotate(
            label,
            (points[idx, 0], points[idx, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=9,
        )
    ax.set_title("PCA of Pooled Sequence Embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.show()


def _save_similarity_heatmaps(
    embeddings: np.ndarray,
    labels: list[str],
    output_path: str,
) -> None:
    """Save cosine and Euclidean similarity heatmaps.

    :param embeddings: Pooled embeddings of shape (N, D).
    :param labels: Labels of length N.
    :param output_path: Output image path.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N, D)")
    if embeddings.shape[0] != len(labels):
        raise ValueError("labels must match embeddings length")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    normed = embeddings / safe_norms
    cosine_sim = normed @ normed.T

    diffs = embeddings[:, None, :] - embeddings[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    euclid_sim = 1.0 / (1.0 + dist)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cos_im = axes[0].imshow(cosine_sim, cmap="viridis", vmin=-1.0, vmax=1.0)
    axes[0].set_title("Cosine Similarity")
    fig.colorbar(cos_im, ax=axes[0], fraction=0.046, pad=0.04)

    euclid_im = axes[1].imshow(euclid_sim, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("Euclidean Similarity (1 / (1 + d))")
    fig.colorbar(euclid_im, ax=axes[1], fraction=0.046, pad=0.04)

    tick_positions = np.arange(len(labels))
    for ax in axes:
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _render_token_html(
    labels: list[str],
    token_ids_per_sample: list[list[int]],
    colors_per_token: list[list[tuple[int, int, int]]],
    *,
    tokenizer_name: str,
    eos_token: str,
    pad_token: str,
    mask_token: str,
) -> str:
    """Render an HTML document with per-token background colors.

    :param labels: Label strings per sample.
    :param token_ids_per_sample: Token ids per sample.
    :param colors_per_token: RGB colors per token for each sample.
    :param tokenizer_name: Tokenizer identifier.
    :param eos_token: EOS token string.
    :param pad_token: Padding token string.
    :param mask_token: Masking token string.
    :returns: HTML string.
    """
    if len(labels) != len(token_ids_per_sample):
        raise ValueError("labels and token lists must match")
    if len(labels) != len(colors_per_token):
        raise ValueError("labels and colors must match")

    tokenizer, _eos_id, _pad_id, _mask_id = _build_tokenizer(
        tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
    )

    sections: list[str] = []
    for label, token_ids, colors in zip(labels, token_ids_per_sample, colors_per_token, strict=True):
        if len(token_ids) != len(colors):
            raise ValueError("token count must match color count")
        spans: list[str] = []
        for token_id, color in zip(token_ids, colors, strict=True):
            decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            if len(decoded) == 0:
                decoded = eos_token
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "#000000" if luminance >= 128.0 else "#ffffff"
            spans.append(
                "<span class=\"token\" "
                f"style=\"background-color: rgb({r}, {g}, {b}); color: {text_color};\">"
                f"{decoded}</span>"
            )
        section = (
            f"<h2>{label}</h2>"
            "<div class=\"container\">"
            f"{''.join(spans)}"
            "</div>"
        )
        sections.append(section)

    body = "\n".join(sections)
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
    h2 {{
      margin: 24px 0 8px;
      font-size: 16px;
    }}
    .container {{
      white-space: pre-wrap;
      line-height: 1.5;
      font-size: 14px;
      margin-bottom: 12px;
    }}
    .token {{
      padding: 0 2px;
      border-radius: 2px;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    """Run the sequence embedding PCA plot."""
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

    entries = _load_yaml_entries(args.yaml_path)
    tokens, attention_mask, labels, token_ids_per_sample = _prepare_batch(
        entries,
        tokenizer_name=tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
        max_seq_len=max_seq_len,
    )

    model = _load_model(checkpoint_dir, dtype=dtype, model_config=model_config)
    token_tensor = jnp.asarray(tokens, dtype=jnp.int32)
    mask_tensor = jnp.asarray(attention_mask, dtype=jnp.bool_)
    token_reps, pooled = model(token_tensor, mask_tensor, train=False, key=None)
    pooled = jax.device_get(pooled)
    pooled_np = np.asarray(pooled, dtype=np.float32)
    token_reps = jax.device_get(token_reps)
    token_reps_np = np.asarray(token_reps, dtype=np.float32)

    mask_flat = attention_mask.reshape(-1)
    reps_flat = token_reps_np.reshape((-1, token_reps_np.shape[-1]))
    valid_reps = reps_flat[mask_flat]
    token_colors_flat = _pca_to_rgb(valid_reps)

    colors_per_sample: list[list[tuple[int, int, int]]] = []
    flat_index = 0
    for token_ids in token_ids_per_sample:
        num_tokens = len(token_ids)
        if num_tokens == 0:
            colors_per_sample.append([])
            continue
        slice_colors = token_colors_flat[flat_index : flat_index + num_tokens]
        colors = [tuple(map(int, color)) for color in slice_colors]
        colors_per_sample.append(colors)
        flat_index += num_tokens

    if flat_index != token_colors_flat.shape[0]:
        raise ValueError("token color assignment mismatch")

    points = _pca_2d(pooled_np)
    _plot_embeddings(points, labels)

    base_dir = os.path.dirname(os.path.abspath(args.yaml_path))
    stem = os.path.splitext(os.path.basename(args.yaml_path))[0]
    heatmap_path = os.path.join(base_dir, f"{stem}_similarity.png")
    html_path = os.path.join(base_dir, f"{stem}_tokens.html")

    _save_similarity_heatmaps(pooled_np, labels, heatmap_path)
    html_output = _render_token_html(
        labels,
        token_ids_per_sample,
        colors_per_sample,
        tokenizer_name=tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
    )
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(html_output)
    print(f"Wrote similarity heatmaps to: {heatmap_path}")
    print(f"Wrote token PCA HTML to: {html_path}")


if __name__ == "__main__":
    main()
