import argparse
import json
import os
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
        description="Compare pooled embeddings for three texts from a checkpointed model."
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory or model.eqx file.")
    parser.add_argument("anchor", type=str, help="Anchor text.")
    parser.add_argument("candidate_a", type=str, help="First candidate text.")
    parser.add_argument("candidate_b", type=str, help="Second candidate text.")
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
) -> tuple[list[int], int, int]:
    """Tokenize text the same way as the streaming dataset.

    :param text: Input text string.
    :param tokenizer_name: Hugging Face tokenizer identifier.
    :param eos_token: EOS token string.
    :param pad_token: Padding token string.
    :param mask_token: Masking token string.
    :param max_seq_len: Maximum sequence length.
    :returns: Tuple of (token ids, pad token id, eos token id).
    """
    _tokenizer, eos_id, pad_id, _mask_id = _build_tokenizer(
        tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
    )
    if max_seq_len <= 1:
        raise ValueError("max_seq_len must be > 1")
    encoding = _tokenizer.encode(text, add_special_tokens=False)
    token_ids = list(encoding.ids)
    if len(token_ids) > max_seq_len - 1:
        token_ids = token_ids[: max_seq_len - 1]
    token_ids.append(eos_id)
    return token_ids, pad_id, eos_id


def _prepare_batch(
    token_ids: list[int],
    *,
    max_seq_len: int,
    pad_id: int,
    eos_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad token ids into a batch and build the attention mask.

    :param token_ids: Token ids for a single sample.
    :param max_seq_len: Maximum sequence length.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
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
    eos_positions = tokens == eos_id
    if eos_positions.any():
        tokens = np.where(eos_positions, pad_id, tokens)
        attention_mask = np.where(eos_positions, False, attention_mask)
    return tokens, attention_mask


def _pooled_embedding(
    model: TextTransformer,
    token_ids: list[int],
    *,
    max_seq_len: int,
    pad_id: int,
    eos_id: int,
    dtype: jnp.dtype,
) -> np.ndarray:
    """Compute pooled, normalized embedding for a single sequence.

    :param model: Text transformer model.
    :param token_ids: Token ids for a single sample.
    :param max_seq_len: Maximum sequence length.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :param dtype: Compute dtype.
    :returns: Pooled embedding array of shape (d_model,).
    """
    tokens, attention_mask = _prepare_batch(
        token_ids,
        max_seq_len=max_seq_len,
        pad_id=pad_id,
        eos_id=eos_id,
    )
    token_tensor = jnp.asarray(tokens, dtype=jnp.int32)
    mask_tensor = jnp.asarray(attention_mask, dtype=jnp.bool_)
    _token_reps, pooled = model(token_tensor, mask_tensor, train=False, key=None)
    pooled = jax.device_get(pooled)
    return np.asarray(pooled, dtype=np.float32)[0]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    :param a: First vector of shape (D,).
    :param b: Second vector of shape (D,).
    :returns: Cosine similarity.
    """
    if a.shape != b.shape:
        raise ValueError("cosine similarity requires matching shapes")
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise ValueError("cosine similarity undefined for zero-norm vector")
    return float(np.dot(a, b) / (norm_a * norm_b))


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors.

    :param a: First vector of shape (D,).
    :param b: Second vector of shape (D,).
    :returns: Euclidean distance.
    """
    if a.shape != b.shape:
        raise ValueError("euclidean distance requires matching shapes")
    return float(np.linalg.norm(a - b))


def main() -> None:
    """Run the text similarity comparison."""
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

    anchor_ids, pad_id, eos_id = _tokenize_text(
        args.anchor,
        tokenizer_name=tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
        max_seq_len=max_seq_len,
    )
    candidate_a_ids, _pad_id_a, _eos_id_a = _tokenize_text(
        args.candidate_a,
        tokenizer_name=tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
        max_seq_len=max_seq_len,
    )
    candidate_b_ids, _pad_id_b, _eos_id_b = _tokenize_text(
        args.candidate_b,
        tokenizer_name=tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
        max_seq_len=max_seq_len,
    )

    anchor_vec = _pooled_embedding(
        model,
        anchor_ids,
        max_seq_len=max_seq_len,
        pad_id=pad_id,
        eos_id=eos_id,
        dtype=dtype,
    )
    candidate_a_vec = _pooled_embedding(
        model,
        candidate_a_ids,
        max_seq_len=max_seq_len,
        pad_id=pad_id,
        eos_id=eos_id,
        dtype=dtype,
    )
    candidate_b_vec = _pooled_embedding(
        model,
        candidate_b_ids,
        max_seq_len=max_seq_len,
        pad_id=pad_id,
        eos_id=eos_id,
        dtype=dtype,
    )

    cos_a = _cosine_similarity(anchor_vec, candidate_a_vec)
    cos_b = _cosine_similarity(anchor_vec, candidate_b_vec)
    dist_a = _euclidean_distance(anchor_vec, candidate_a_vec)
    dist_b = _euclidean_distance(anchor_vec, candidate_b_vec)

    print("Anchor vs Candidate A:")
    print(f"  Cosine similarity: {cos_a:.6f}")
    print(f"  Euclidean distance: {dist_a:.6f}")
    print("Anchor vs Candidate B:")
    print(f"  Cosine similarity: {cos_b:.6f}")
    print(f"  Euclidean distance: {dist_b:.6f}")


if __name__ == "__main__":
    main()
