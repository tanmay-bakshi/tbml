import argparse
import json
import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

from tbml.experiments.honeycomb.text.dataset import _build_tokenizer
from tbml.experiments.honeycomb.text.model import TextTransformer, TextTransformerConfig


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Plot PCA of token embeddings for colors, cities, and people."
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory or model.eqx file.")
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
    model_key = jax.random.PRNGKey(0)
    model = TextTransformer(
        config,
        dtype=dtype,
        param_dtype=dtype,
        key=model_key,
    )
    model_path = os.path.join(checkpoint_dir, "model.eqx")
    return eqx.tree_deserialise_leaves(model_path, model)


def _token_id(tokenizer: Tokenizer, text: str) -> int:
    """Get the single-token id for a string.

    :param tokenizer: Tokenizer instance.
    :param text: Input text (with leading space if needed).
    :returns: Token id.
    """
    encoding = tokenizer.encode(text, add_special_tokens=False)
    if len(encoding.ids) != 1:
        raise ValueError(f"token '{text}' does not map to a single token")
    return int(encoding.ids[0])


def _collect_tokens(tokenizer: Tokenizer, vocab_size: int) -> tuple[list[str], list[str], list[int]]:
    """Collect token labels, categories, and ids.

    :param tokenizer: Tokenizer instance.
    :param vocab_size: Vocabulary size for validation.
    :returns: Tuple of (labels, categories, token ids).
    """
    colors = [" red", " blue", " green", " yellow"]
    cities = [" London", " Paris", " Tokyo", " Berlin"]
    people = [" John", " Mary", " Alice", " Bob"]

    labels: list[str] = []
    categories: list[str] = []
    token_ids: list[int] = []

    for group_name, items in (("color", colors), ("city", cities), ("person", people)):
        for item in items:
            token_id = _token_id(tokenizer, item)
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(f"token id out of range for '{item}': {token_id}")
            labels.append(item.lstrip())
            categories.append(group_name)
            token_ids.append(token_id)

    return labels, categories, token_ids


def _pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D via PCA.

    :param embeddings: Token embeddings of shape (N, D).
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


def _plot_embeddings(points: np.ndarray, labels: list[str], categories: list[str]) -> None:
    """Render a labelled scatter plot.

    :param points: PCA points of shape (N, 2).
    :param labels: Token labels of length N.
    :param categories: Category names of length N.
    """
    if points.shape[0] != len(labels) or points.shape[0] != len(categories):
        raise ValueError("points, labels, and categories must have matching lengths")

    palette = {
        "color": "#f94144",
        "city": "#277da1",
        "person": "#43aa8b",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for group_name in ("color", "city", "person"):
        indices = [idx for idx, group in enumerate(categories) if group == group_name]
        if len(indices) == 0:
            continue
        group_points = points[indices]
        ax.scatter(
            group_points[:, 0],
            group_points[:, 1],
            s=70,
            label=group_name,
            color=palette.get(group_name, "#222222"),
        )
        for idx in indices:
            ax.annotate(
                labels[idx],
                (points[idx, 0], points[idx, 1]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=9,
            )

    ax.set_title("PCA of GPT-2 Token Embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the token embedding PCA plot."""
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
    if tokenizer_name is None or isinstance(tokenizer_name, str) is False:
        raise ValueError("data config missing tokenizer name")
    if tokenizer_name != "gpt2":
        raise ValueError("this script expects the tokenizer to be gpt2")
    if eos_token is None or isinstance(eos_token, str) is False:
        raise ValueError("data config missing eos token")
    if pad_token is None or isinstance(pad_token, str) is False:
        raise ValueError("data config missing pad token")
    if mask_token is None or isinstance(mask_token, str) is False:
        raise ValueError("data config missing mask token")

    model = _load_model(checkpoint_dir, dtype=dtype, model_config=model_config)
    tokenizer, _eos_id, _pad_id, _mask_id = _build_tokenizer(
        tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
    )

    vocab_size = model.config.vocab_size
    labels, categories, token_ids = _collect_tokens(tokenizer, vocab_size)
    token_array = jnp.asarray(token_ids, dtype=jnp.int32)
    embeddings = jnp.take(model.token_embed.weight, token_array, axis=0)
    embeddings = jax.device_get(embeddings)
    embeddings_np = np.asarray(embeddings, dtype=np.float32)
    points = _pca_2d(embeddings_np)

    _plot_embeddings(points, labels, categories)


if __name__ == "__main__":
    main()
