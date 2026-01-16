import argparse
import json
import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tbml.experiments.honeycomb.text.dataset import _build_tokenizer
from tbml.experiments.honeycomb.text.model import TextTransformer, TextTransformerConfig


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Visualize the embedding trajectory of masked-prefix samples."
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory or model.eqx file.")
    parser.add_argument("text", type=str, help="Input text to tokenize.")
    parser.add_argument("--pca-dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--output", type=str, default="trajectory.png")
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
    raise RuntimeError("failed to load model")


def _pca(embeddings: np.ndarray, *, dim: int) -> np.ndarray:
    """Project embeddings via PCA.

    :param embeddings: Embeddings of shape (N, D).
    :param dim: Number of PCA dimensions.
    :returns: PCA scores of shape (N, dim).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N, D)")
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, v_t = np.linalg.svd(centered, full_matrices=False)
    components = v_t[: min(dim, v_t.shape[0]), :]
    scores = centered @ components.T
    if scores.shape[1] < dim:
        pad = np.zeros((scores.shape[0], dim - scores.shape[1]), dtype=scores.dtype)
        scores = np.concatenate([scores, pad], axis=1)
    return scores


def _plot_trajectory_2d(points: np.ndarray, *, title: str, output: str, interactive: bool) -> None:
    """Plot 2D trajectory with arrows.

    :param points: PCA points of shape (N, 2).
    :param title: Plot title.
    :param output: Output path for saved figure.
    :param interactive: Whether to display interactively.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(points[:, 0], points[:, 1], marker="o", color="#277da1", linewidth=1.5)
    for idx in range(points.shape[0] - 1):
        ax.annotate(
            "",
            xy=(points[idx + 1, 0], points[idx + 1, 1]),
            xytext=(points[idx, 0], points[idx, 1]),
            arrowprops=dict(arrowstyle="->", color="#577590", linewidth=1.0),
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    if interactive:
        plt.show()
    else:
        plt.savefig(output, dpi=200)
        print(f"Wrote trajectory plot to: {output}")


def _plot_trajectory_3d(points: np.ndarray, *, title: str, output: str, interactive: bool) -> None:
    """Plot 3D trajectory with arrows.

    :param points: PCA points of shape (N, 3).
    :param title: Plot title.
    :param output: Output path for saved figure.
    :param interactive: Whether to display interactively.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points[:, 0], points[:, 1], points[:, 2], marker="o", color="#277da1", linewidth=1.2)
    for idx in range(points.shape[0] - 1):
        start = points[idx]
        delta = points[idx + 1] - points[idx]
        ax.quiver(
            start[0],
            start[1],
            start[2],
            delta[0],
            delta[1],
            delta[2],
            arrow_length_ratio=0.15,
            color="#577590",
            linewidth=1.0,
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    if interactive:
        plt.show()
    else:
        plt.savefig(output, dpi=200)
        print(f"Wrote trajectory plot to: {output}")


def _build_samples(
    token_ids: list[int],
    *,
    mask_id: int,
    eos_id: int,
    pad_id: int,
    max_seq_len: int,
) -> np.ndarray:
    """Build masked-prefix samples from a token sequence (excluding EOS).

    :param token_ids: Token ids without EOS.
    :param mask_id: Mask token id.
    :param eos_id: EOS token id.
    :param pad_id: Padding token id.
    :param max_seq_len: Maximum sequence length.
    :returns: Array of shape (N, max_seq_len).
    """
    if len(token_ids) == 0:
        raise ValueError("token_ids must be non-empty")
    samples: list[np.ndarray] = []
    for idx in range(len(token_ids)):
        prefix = list(token_ids[: idx + 1])
        max_prefix = max_seq_len - 2
        if len(prefix) > max_prefix:
            prefix = prefix[:max_prefix]
        sample_tokens = prefix + [mask_id, eos_id]
        sample = np.full((max_seq_len,), pad_id, dtype=np.int32)
        sample[: len(sample_tokens)] = np.asarray(sample_tokens, dtype=np.int32)
        samples.append(sample)
    return np.stack(samples, axis=0)


def main() -> None:
    """Run the trajectory visualization."""
    args = _parse_args()
    if args.pca_dim not in (2, 3):
        raise ValueError("pca-dim must be 2 or 3")
    if args.interactive is False and args.output == "":
        raise ValueError("output path required when not using --interactive")

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
    if isinstance(dtype_name, str) is False:
        raise ValueError("training config missing dtype string")

    tokenizer_name = data_config.get("tokenizer")
    eos_token = data_config.get("eos_token")
    pad_token = data_config.get("pad_token")
    mask_token = data_config.get("mask_token")
    max_seq_len = data_config.get("max_seq_len")
    if isinstance(tokenizer_name, str) is False:
        raise ValueError("data config missing tokenizer name")
    if isinstance(eos_token, str) is False:
        raise ValueError("data config missing eos token")
    if isinstance(pad_token, str) is False:
        raise ValueError("data config missing pad token")
    if isinstance(mask_token, str) is False:
        raise ValueError("data config missing mask token")
    if isinstance(max_seq_len, int) is False:
        raise ValueError("data config missing max_seq_len")

    dtype = _dtype_from_name(dtype_name)
    tokenizer, eos_id, pad_id, mask_id = _build_tokenizer(
        tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
    )

    encoding = tokenizer.encode(args.text, add_special_tokens=False)
    token_ids = list(encoding.ids)
    if len(token_ids) > max_seq_len - 1:
        token_ids = token_ids[: max_seq_len - 1]
    token_ids_no_eos = token_ids

    samples = _build_samples(
        token_ids_no_eos,
        mask_id=int(mask_id),
        eos_id=int(eos_id),
        pad_id=int(pad_id),
        max_seq_len=max_seq_len,
    )

    model = _load_model(checkpoint_dir, dtype=dtype, model_config=model_config)
    tokens = jnp.asarray(samples)
    attention_mask = jnp.asarray(samples != int(pad_id))
    embeddings = model(tokens, attention_mask, train=False, key=None)
    embeddings = np.asarray(embeddings)

    points = _pca(embeddings, dim=args.pca_dim)
    title = "Trajectory of Masked-Prefix Embeddings"
    if args.pca_dim == 2:
        _plot_trajectory_2d(points, title=title, output=args.output, interactive=args.interactive)
    else:
        _plot_trajectory_3d(points, title=title, output=args.output, interactive=args.interactive)


if __name__ == "__main__":
    main()
