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
        description="Compare tokenwise reconstruction MSE against a masked reference string."
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory or model.eqx file.")
    parser.add_argument(
        "texts",
        type=str,
        nargs="+",
        help="Input texts: first is the masked reference, remaining are candidates.",
    )
    parser.add_argument(
        "--candidates-use-swa",
        action="store_true",
        help="Run candidate texts through the EMA teacher weights while keeping the reference on the base weights.",
    )
    parser.add_argument(
        "--reference-use-swa",
        action="store_true",
        help="Run the reference text through the EMA teacher weights instead of the base weights.",
    )
    parser.add_argument(
        "--masked-only",
        action="store_true",
        help="Restrict MSE computation to masked positions in the reference.",
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


def _load_swa_model(
    checkpoint_dir: str,
    *,
    dtype: jnp.dtype,
    model_config: dict[str, Any],
) -> TextTransformer:
    """Load the SWA parameters and combine them with static model state.

    :param checkpoint_dir: Checkpoint directory path.
    :param dtype: Compute and parameter dtype.
    :param model_config: Model configuration dictionary.
    :returns: TextTransformer with SWA parameters.
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
            swa_params = eqx.tree_deserialise_leaves(model_path, params)
            return eqx.combine(swa_params, static)
        except RuntimeError as exc:
            last_error = exc
            if "changed dtype" not in str(exc):
                raise
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("failed to load swa parameters")


def _load_model(
    checkpoint_dir: str,
    *,
    dtype: jnp.dtype,
    model_config: dict[str, Any],
) -> TextTransformer:
    """Load the base TextTransformer model from a checkpoint.

    :param checkpoint_dir: Checkpoint directory path.
    :param dtype: Compute and parameter dtype.
    :param model_config: Model configuration dictionary.
    :returns: Deserialized TextTransformer model.
    """
    return _load_model_from_path(
        checkpoint_dir,
        dtype=dtype,
        model_config=model_config,
        filename="model.eqx",
    )


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
    """Run tokenwise MSE comparison for candidate texts."""
    args = _parse_args()
    if len(args.texts) < 2:
        raise ValueError("provide at least one reference text and one candidate text")

    ckpt_dir = _resolve_checkpoint_dir(args.checkpoint)
    run_config = _load_run_config(ckpt_dir)
    model_config = run_config.get("model")
    data_config = run_config.get("data")
    view_config = run_config.get("views")
    training_config = run_config.get("training")
    if isinstance(model_config, dict) is False or isinstance(data_config, dict) is False:
        raise ValueError("checkpoint config is missing model or data section")
    if isinstance(view_config, dict) is False:
        raise ValueError("checkpoint config is missing views section")
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
    use_mask_token = view_config.get("mask_token_input", False)
    if isinstance(use_mask_token, bool) is False:
        raise ValueError("mask_token_input must be a boolean if present in the config")
    if use_mask_token is True:
        ref_attn_mask = ref_attn
    else:
        ref_attn_mask = np.logical_and(ref_attn, np.logical_not(mask_positions))

    model = _load_model(ckpt_dir, dtype=dtype, model_config=model_config)
    swa_path = os.path.join(ckpt_dir, "swa.eqx")
    if os.path.isfile(swa_path) is True:
        if args.reference_use_swa is False and args.candidates_use_swa is False:
            args.reference_use_swa = True
            args.candidates_use_swa = True
    swa_model: TextTransformer | None = None
    if args.candidates_use_swa is True or args.reference_use_swa is True:
        swa_model = _load_swa_model(
            ckpt_dir,
            dtype=dtype,
            model_config=model_config,
        )
    ref_model = swa_model if args.reference_use_swa else model
    cand_model = swa_model if args.candidates_use_swa else model
    _ref_pre, ref_post, _ref_pool = ref_model.forward_with_intermediates(
        jnp.asarray(ref_tokens_arr),
        jnp.asarray(ref_attn_mask),
        train=False,
        key=None,
    )
    if ref_model.predictor is None:
        pred_reps = ref_post
    else:
        pred_attn = np.logical_or(ref_attn_mask, mask_positions)
        pred_reps = ref_model.predictor(
            ref_post,
            jnp.asarray(pred_attn),
            jnp.asarray(mask_positions),
            train=False,
            key=None,
        )

    _cand_pre, cand_post, _cand_pool = cand_model.forward_with_intermediates(
        jnp.asarray(cand_tokens_arr),
        jnp.asarray(cand_attn),
        train=False,
        key=None,
    )

    if args.masked_only is True:
        rec_mask = np.logical_and(mask_positions, ref_tokens_arr != pad_id)
    else:
        rec_mask = np.logical_or(ref_attn_mask, mask_positions)
    pred_mse = _masked_mean_square_error(pred_reps[0], cand_post, jnp.asarray(rec_mask[0]))
    pred_mse = np.asarray(jax.device_get(pred_mse))
    enc_mse = _masked_mean_square_error(ref_post[0], cand_post, jnp.asarray(rec_mask[0]))
    enc_mse = np.asarray(jax.device_get(enc_mse))

    pred_results = list(zip(args.texts[1:], pred_mse, strict=True))
    pred_results.sort(key=lambda item: float(item[1]))
    enc_results = list(zip(args.texts[1:], enc_mse, strict=True))
    enc_results.sort(key=lambda item: float(item[1]))

    print("Predictor vs candidate encoder MSE (ascending):")
    for idx, (text, score) in enumerate(pred_results, start=1):
        print(f"  {idx:>2}. {score:.6f} :: {repr(text)}")
    print("Reference encoder vs candidate encoder MSE (ascending):")
    for idx, (text, score) in enumerate(enc_results, start=1):
        print(f"  {idx:>2}. {score:.6f} :: {repr(text)}")


if __name__ == "__main__":
    main()
