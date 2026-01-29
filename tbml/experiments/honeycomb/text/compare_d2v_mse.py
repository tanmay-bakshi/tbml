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
        description="Compare candidates using the data2vec objective."
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory or model.eqx file.")
    parser.add_argument(
        "texts",
        type=str,
        nargs="+",
        help="Input texts: first is the masked reference, remaining are candidates.",
    )
    parser.add_argument(
        "--teacher-use-base",
        action="store_true",
        help="Use the base model for teacher targets even if EMA weights exist.",
    )
    parser.add_argument(
        "--student-use-swa",
        action="store_true",
        help="Use EMA teacher weights as the student model.",
    )
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


def _load_model_from_path(
    checkpoint_dir: str,
    *,
    dtype: jnp.dtype,
    model_config: dict[str, Any],
    filename: str,
) -> TextTransformer:
    """Load a TextTransformer model from a checkpoint.

    :param checkpoint_dir: Checkpoint directory path.
    :param dtype: Compute and parameter dtype.
    :param model_config: Model configuration dictionary.
    :param filename: Checkpoint filename to load.
    :returns: Deserialized TextTransformer model.
    """
    config = TextTransformerConfig(**model_config)
    model_path = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(model_path) is False:
        raise FileNotFoundError(f"{filename} not found in checkpoint directory: {checkpoint_dir}")
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


def _load_teacher_model(
    checkpoint_dir: str,
    *,
    dtype: jnp.dtype,
    model_config: dict[str, Any],
) -> TextTransformer:
    """Load the teacher parameters and combine them with static model state.

    :param checkpoint_dir: Checkpoint directory path.
    :param dtype: Compute and parameter dtype.
    :param model_config: Model configuration dictionary.
    :returns: TextTransformer with teacher parameters.
    """
    config = TextTransformerConfig(**model_config)
    model_path = os.path.join(checkpoint_dir, "swa.eqx")
    if os.path.isfile(model_path) is False:
        raise FileNotFoundError(f"swa.eqx not found in checkpoint directory: {checkpoint_dir}")

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
        params, static = eqx.partition(model, eqx.is_array)
        try:
            teacher_params = eqx.tree_deserialise_leaves(model_path, params)
            return eqx.combine(teacher_params, static)
        except RuntimeError as exc:
            last_error = exc
            if "changed dtype" not in str(exc):
                raise
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("failed to load teacher parameters")


def _tokenize_texts(
    texts: list[str],
    *,
    tokenizer_name: str,
    eos_token: str,
    pad_token: str,
    mask_token: str,
    max_seq_len: int,
) -> tuple[list[list[int]], int, int, int]:
    """Tokenize texts with the dataset preprocessing rules.

    :param texts: List of input texts.
    :param tokenizer_name: Hugging Face tokenizer identifier.
    :param eos_token: EOS token string.
    :param pad_token: Padding token string.
    :param mask_token: Mask token string.
    :param max_seq_len: Maximum sequence length.
    :returns: Tuple of (token lists, pad id, eos id, mask id).
    """
    tokenizer, eos_id, pad_id, mask_id = _build_tokenizer(
        tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
    )
    token_lists: list[list[int]] = []
    for text in texts:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        token_ids = list(encoding.ids)
        if len(token_ids) > max_seq_len - 1:
            token_ids = token_ids[: max_seq_len - 1]
        token_ids.append(eos_id)
        token_lists.append(token_ids)
    return token_lists, pad_id, eos_id, mask_id


def _prepare_batch(
    token_lists: list[list[int]],
    *,
    max_seq_len: int,
    pad_id: int,
    eos_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad token lists into a batch and build attention masks.

    :param token_lists: Token ids for each sample.
    :param max_seq_len: Maximum sequence length.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :returns: Tuple of (tokens, attention_mask) arrays.
    """
    batch = len(token_lists)
    tokens = np.full((batch, max_seq_len), pad_id, dtype=np.int32)
    attention = np.zeros((batch, max_seq_len), dtype=np.bool_)
    for idx, token_ids in enumerate(token_lists):
        if len(token_ids) > max_seq_len:
            raise ValueError("tokenized length exceeds max_seq_len")
        tokens[idx, : len(token_ids)] = np.asarray(token_ids, dtype=np.int32)
        attention[idx, : len(token_ids)] = True
    eos_positions = tokens == eos_id
    if eos_positions.any():
        tokens = np.where(eos_positions, pad_id, tokens)
        attention = np.where(eos_positions, False, attention)
    return tokens, attention


def _instance_norm(
    reps: Array,
    mask: Array,
    *,
    eps: float = 1e-5,
) -> Array:
    """Apply instance normalization over the sequence dimension.

    :param reps: Representations of shape (B, T, D).
    :param mask: Boolean mask of shape (B, T) for valid positions.
    :param eps: Numerical stability epsilon.
    :returns: Instance-normalized reps of shape (B, T, D).
    """
    if reps.ndim != 3:
        raise ValueError("reps must have shape (B, T, D)")
    if mask.ndim != 2:
        raise ValueError("mask must have shape (B, T)")
    if reps.shape[:2] != mask.shape:
        raise ValueError("reps and mask must align on (B, T)")

    reps_f = reps.astype(jnp.float32)
    mask_f = mask.astype(jnp.float32)
    count = jnp.sum(mask_f, axis=1, keepdims=True)
    count = jnp.maximum(count, 1.0)
    mean = jnp.sum(reps_f * mask_f[:, :, None], axis=1, keepdims=True) / count[:, None]
    var = jnp.sum(jnp.square(reps_f - mean) * mask_f[:, :, None], axis=1, keepdims=True) / count[:, None]
    return (reps_f - mean) / jnp.sqrt(var + eps)


def _teacher_targets(
    model: TextTransformer,
    tokens: Array,
    attention_mask: Array,
    *,
    top_k: int,
    use_instance_norm: bool,
    eps: float = 1e-5,
) -> Array:
    """Compute data2vec teacher targets from top-K encoder blocks.

    :param model: Teacher model.
    :param tokens: Token ids of shape (B, T).
    :param attention_mask: Boolean mask of shape (B, T) for valid positions.
    :param top_k: Number of top FFN blocks to average.
    :param use_instance_norm: Whether to instance-normalize layer outputs before averaging.
    :param eps: Instance norm epsilon.
    :returns: Target representations of shape (B, T, D) in float32.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (B, T)")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape (B, T)")
    if tokens.shape != attention_mask.shape:
        raise ValueError("tokens and attention_mask must have the same shape")
    if top_k > len(model.blocks):
        raise ValueError("top_k must be <= number of encoder blocks")

    reps = model.token_embed(tokens)
    outputs: list[Array] = []
    for block in model.blocks:
        reps = block(reps, attention_mask=attention_mask, train=False, key=None)
        outputs.append(reps)
    selected = outputs[-top_k:]
    if use_instance_norm is True:
        normed = [_instance_norm(layer_out, attention_mask, eps=eps) for layer_out in selected]
        stacked = jnp.stack(normed, axis=0)
    else:
        stacked = jnp.stack(selected, axis=0)
    return jnp.mean(stacked, axis=0)


def _masked_mean_square_error(
    ref_reps: Array,
    cand_reps: Array,
    mask: Array,
) -> Array:
    """Compute masked tokenwise MSE.

    :param ref_reps: Reference reps of shape (T, D).
    :param cand_reps: Candidate reps of shape (N, T, D).
    :param mask: Mask of shape (T,).
    :returns: MSE values of shape (N,).
    """
    if ref_reps.ndim != 2:
        raise ValueError("ref_reps must have shape (T, D)")
    if cand_reps.ndim != 3:
        raise ValueError("cand_reps must have shape (N, T, D)")
    if mask.ndim != 1:
        raise ValueError("mask must have shape (T,)")
    if ref_reps.shape[0] != cand_reps.shape[1] or ref_reps.shape[0] != mask.shape[0]:
        raise ValueError("sequence length mismatch")

    diffs = cand_reps.astype(jnp.float32) - ref_reps.astype(jnp.float32)[None, :, :]
    mse = jnp.mean(jnp.square(diffs), axis=-1)
    mask_f = mask.astype(jnp.float32)
    loss_sum = jnp.sum(mse * mask_f[None, :], axis=1)
    count = jnp.sum(mask_f)
    return jnp.where(count > 0.0, loss_sum / count, 0.0)


def main() -> None:
    """Run data2vec-style candidate ranking."""
    args = _parse_args()
    if len(args.texts) < 2:
        raise ValueError("provide at least one reference text and one candidate text")

    ckpt_dir = _resolve_checkpoint_dir(args.checkpoint)
    run_config = _load_run_config(ckpt_dir)
    model_config = run_config.get("model")
    data_config = run_config.get("data")
    loss_config = run_config.get("loss")
    training_config = run_config.get("training")
    if isinstance(model_config, dict) is False or isinstance(data_config, dict) is False:
        raise ValueError("checkpoint config is missing model or data section")
    if isinstance(loss_config, dict) is False:
        raise ValueError("checkpoint config is missing loss section")
    if isinstance(training_config, dict) is False:
        raise ValueError("checkpoint config is missing training section")

    dtype_name = training_config.get("dtype", "float32")
    if isinstance(dtype_name, str) is False:
        raise ValueError("training dtype must be a string")
    dtype = _dtype_from_name(dtype_name)

    tokenizer_name = data_config.get("tokenizer")
    eos_token = data_config.get("eos_token")
    pad_token = data_config.get("pad_token")
    mask_token = data_config.get("mask_token")
    max_seq_len = model_config.get("max_seq_len")
    if isinstance(tokenizer_name, str) is False:
        raise ValueError("tokenizer name missing from config")
    if isinstance(eos_token, str) is False:
        raise ValueError("eos_token missing from config")
    if isinstance(pad_token, str) is False:
        raise ValueError("pad_token missing from config")
    if isinstance(mask_token, str) is False:
        raise ValueError("mask_token missing from config")
    if isinstance(max_seq_len, int) is False:
        raise ValueError("max_seq_len missing from config")

    top_k = loss_config.get("teacher_top_k")
    if isinstance(top_k, int) is False:
        raise ValueError("teacher_top_k missing from config")
    use_instance_norm = loss_config.get("teacher_instance_norm", True)
    if isinstance(use_instance_norm, bool) is False:
        raise ValueError("teacher_instance_norm must be a boolean")

    token_lists, pad_id, eos_id, mask_id = _tokenize_texts(
        list(args.texts),
        tokenizer_name=tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
        max_seq_len=max_seq_len,
    )
    lengths = [len(tokens) for tokens in token_lists]
    if len(set(lengths)) != 1:
        raise ValueError("all texts must tokenize to the same length")

    ref_tokens = token_lists[0]
    cand_tokens = token_lists[1:]
    if any(mask_id in tokens for tokens in cand_tokens):
        raise ValueError("candidate texts must not include mask tokens")

    tokens_all, attn_all = _prepare_batch(
        token_lists,
        max_seq_len=max_seq_len,
        pad_id=pad_id,
        eos_id=eos_id,
    )
    ref_tokens_arr = tokens_all[:1]
    cand_tokens_arr = tokens_all[1:]
    ref_attn = attn_all[:1]
    cand_attn = attn_all[1:]

    mask_positions = ref_tokens_arr == mask_id
    if not np.any(mask_positions):
        raise ValueError("reference text must include at least one mask token")

    ref_attn_mask = np.logical_and(ref_attn, np.logical_not(mask_positions))

    model = _load_model_from_path(
        ckpt_dir,
        dtype=dtype,
        model_config=model_config,
        filename="model.eqx",
    )
    swa_path = os.path.join(ckpt_dir, "swa.eqx")
    if os.path.isfile(swa_path) is True and args.teacher_use_base is False:
        teacher_model = _load_teacher_model(
            ckpt_dir,
            dtype=dtype,
            model_config=model_config,
        )
    else:
        teacher_model = model
    if args.student_use_swa is True:
        if os.path.isfile(swa_path) is False:
            raise FileNotFoundError("swa.eqx not found for student-use-swa")
        model = _load_teacher_model(
            ckpt_dir,
            dtype=dtype,
            model_config=model_config,
        )

    _ref_pre, ref_post, _ref_pool = model.forward_with_intermediates(
        jnp.asarray(ref_tokens_arr),
        jnp.asarray(ref_attn_mask),
        train=False,
        key=None,
    )
    if model.predictor is None:
        raise ValueError("predictor is required for data2vec comparison")
    pred_attn = np.logical_or(ref_attn_mask, mask_positions)
    pred_reps = model.predictor(
        ref_post,
        jnp.asarray(pred_attn),
        jnp.asarray(mask_positions),
        train=False,
        key=None,
    )

    tokens_no_eos = np.where(cand_tokens_arr == eos_id, pad_id, cand_tokens_arr)
    teacher_targets = _teacher_targets(
        teacher_model,
        jnp.asarray(tokens_no_eos),
        jnp.asarray(cand_attn),
        top_k=top_k,
        use_instance_norm=use_instance_norm,
    )

    rec_mask = np.logical_and(mask_positions, ref_tokens_arr != pad_id)
    scores = _masked_mean_square_error(pred_reps[0], teacher_targets, jnp.asarray(rec_mask[0]))
    scores = np.asarray(jax.device_get(scores))

    results = list(zip(args.texts[1:], scores, strict=True))
    results.sort(key=lambda item: float(item[1]))
    print("Predictor vs teacher targets MSE (masked positions, ascending):")
    for idx, (text, score) in enumerate(results, start=1):
        print(f"  {idx:>2}. {score:.6f} :: {repr(text)}")


if __name__ == "__main__":
    main()
