import argparse
import json
import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from tbml.experiments.honeycomb.text.dataset import _build_tokenizer
from tbml.experiments.honeycomb.text.model import TextTransformer, TextTransformerConfig


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Score candidate suffixes by predictor/base span reconstruction distance."
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory or model.eqx file.")
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--suffix", type=str, action="append", required=True)
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


def _tokenize(tokenizer, text: str) -> list[int]:
    """Tokenize text without adding special tokens.

    :param tokenizer: Tokenizer instance.
    :param text: Input text.
    :returns: List of token ids.
    """
    encoding = tokenizer.encode(text, add_special_tokens=False)
    return list(encoding.ids)


def _masked_mean(reps: Array, mask: Array) -> Array:
    """Compute the mean over masked positions.

    :param reps: Array of shape (B, T, D).
    :param mask: Boolean mask of shape (B, T).
    :returns: Array of shape (B, D).
    """
    if reps.ndim != 3:
        raise ValueError("reps must have shape (B, T, D)")
    if mask.ndim != 2:
        raise ValueError("mask must have shape (B, T)")
    if reps.shape[:2] != mask.shape:
        raise ValueError("reps and mask must align on (B, T)")
    mask_f = mask.astype(reps.dtype)
    summed = jnp.sum(reps * mask_f[:, :, None], axis=1)
    counts = jnp.sum(mask_f, axis=1)
    counts = jnp.maximum(counts, 1.0)
    return summed / counts[:, None]


def main() -> None:
    """Score candidate suffixes by predictor/base span distance."""
    args = _parse_args()
    if args.suffix is None or len(args.suffix) == 0:
        raise ValueError("at least one --suffix is required")

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

    tokenizer, _eos_id, pad_id, _mask_id = _build_tokenizer(
        tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
    )

    prefix_ids = _tokenize(tokenizer, args.prefix)
    suffix_ids_list = [_tokenize(tokenizer, suffix) for suffix in args.suffix]
    if len(prefix_ids) == 0:
        raise ValueError("prefix must contain at least one token")
    if any(len(suffix_ids) == 0 for suffix_ids in suffix_ids_list):
        raise ValueError("suffixes must contain at least one token")

    suffix_len = len(suffix_ids_list[0])
    if any(len(suffix_ids) != suffix_len for suffix_ids in suffix_ids_list):
        raise ValueError("all suffixes must have the same tokenized length")

    seq_len = len(prefix_ids) + suffix_len
    if seq_len > max_seq_len:
        raise ValueError("prefix + suffix length exceeds max_seq_len")

    model = _load_model(checkpoint_dir, dtype=dtype, model_config=model_config)

    masked_tokens = np.full((1, seq_len), pad_id, dtype=np.int32)
    masked_tokens[0, : len(prefix_ids)] = np.asarray(prefix_ids, dtype=np.int32)
    masked_attn = np.zeros((1, seq_len), dtype=np.bool_)
    masked_attn[0, : len(prefix_ids)] = True
    mask_positions = np.zeros((1, seq_len), dtype=np.bool_)
    mask_positions[0, len(prefix_ids) :] = True

    _token_pre, token_post, _pooled = model.forward_with_intermediates(
        jnp.asarray(masked_tokens),
        jnp.asarray(masked_attn),
        train=False,
        key=None,
    )
    predictor_attn = np.logical_or(masked_attn, mask_positions)
    pred_reps = model.predictor(
        token_post,
        jnp.asarray(predictor_attn),
        jnp.asarray(mask_positions),
        train=False,
        key=None,
    )
    pred_span = _masked_mean(pred_reps, jnp.asarray(mask_positions))
    pred_span = jax.device_get(pred_span)[0]

    num_candidates = len(suffix_ids_list)
    full_tokens = np.zeros((num_candidates, seq_len), dtype=np.int32)
    full_attn = np.zeros((num_candidates, seq_len), dtype=np.bool_)
    for idx, suffix_ids in enumerate(suffix_ids_list):
        tokens = prefix_ids + suffix_ids
        full_tokens[idx, :] = np.asarray(tokens, dtype=np.int32)
        full_attn[idx, :] = True

    _cand_pre, cand_post, _cand_pooled = model.forward_with_intermediates(
        jnp.asarray(full_tokens),
        jnp.asarray(full_attn),
        train=False,
        key=None,
    )
    suffix_mask = np.zeros((num_candidates, seq_len), dtype=np.bool_)
    suffix_mask[:, len(prefix_ids) :] = True
    cand_span = _masked_mean(cand_post, jnp.asarray(suffix_mask))
    cand_span = jax.device_get(cand_span)

    diffs = cand_span - pred_span[None, :]
    mse = np.mean(np.square(np.asarray(diffs)), axis=-1)

    results: list[tuple[int, float, str]] = []
    for idx, (suffix_text, score) in enumerate(zip(args.suffix, mse, strict=True)):
        results.append((idx, float(score), suffix_text))

    results_sorted = sorted(results, key=lambda item: item[1])
    print("Candidate scores (MSE):")
    for rank, (idx, score, suffix_text) in enumerate(results_sorted, start=1):
        print(f"  {rank:>2}. {repr(suffix_text)} -> {score:.6f}")
    best_idx, best_score, best_suffix = results_sorted[0]
    print("")
    print(f"Best: {repr(best_suffix)} (index={best_idx}, mse={best_score:.6f})")


if __name__ == "__main__":
    main()
