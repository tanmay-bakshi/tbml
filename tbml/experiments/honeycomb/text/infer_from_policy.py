import argparse
import json
import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tbml.experiments.honeycomb.text.inference import TextInference
from tbml.experiments.honeycomb.text.train_policy import PolicyTransformer, PolicyTransformerConfig


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Infer next-token predictions from a policy checkpoint.")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--policy-checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
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
    raise ValueError("unsupported dtype")


def _param_dtype_for(dtype: jnp.dtype) -> jnp.dtype:
    """Resolve parameter dtype for a compute dtype.

    :param dtype: Compute dtype.
    :returns: Parameter dtype.
    """
    if dtype in (jnp.bfloat16, jnp.float16):
        return jnp.float32
    return dtype


def _resolve_checkpoint_dir(path: str) -> str:
    """Resolve a checkpoint directory from a user-provided path.

    :param path: Path to a checkpoint directory or a checkpoint file.
    :returns: Resolved checkpoint directory.
    """
    if os.path.isdir(path) is True:
        checkpoint_dir = path
    else:
        checkpoint_dir = os.path.dirname(path)
        if checkpoint_dir == "":
            raise ValueError("checkpoint path must be a directory or a file inside a directory")

    model_path = os.path.join(checkpoint_dir, "model.eqx")
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    if os.path.isfile(model_path) is False:
        raise FileNotFoundError(f"model.eqx not found in checkpoint directory: {checkpoint_dir}")
    if os.path.isfile(metadata_path) is False:
        raise FileNotFoundError(f"metadata.json not found in checkpoint directory: {checkpoint_dir}")
    return checkpoint_dir


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


def _load_policy_model(
    checkpoint_dir: str,
    *,
    dtype: jnp.dtype,
    policy_config: dict[str, Any],
) -> PolicyTransformer:
    """Load a PolicyTransformer model from a checkpoint.

    :param checkpoint_dir: Checkpoint directory path.
    :param dtype: Compute dtype to try first.
    :param policy_config: Policy configuration dictionary.
    :returns: Deserialized PolicyTransformer model.
    """
    config = PolicyTransformerConfig(**policy_config)
    model_path = os.path.join(checkpoint_dir, "model.eqx")
    candidates = [dtype, jnp.float32, jnp.bfloat16, jnp.float16]
    seen: set[jnp.dtype] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        model_key = jax.random.PRNGKey(0)
        model = PolicyTransformer(
            config,
            dtype=candidate,
            param_dtype=_param_dtype_for(candidate),
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
    raise RuntimeError("failed to load policy model")


def main() -> None:
    """Run policy inference on a single text input."""
    args = _parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be > 0")

    base_infer = TextInference.from_checkpoint(args.base_checkpoint)
    tokens, attention_mask = base_infer.preprocess([args.text])
    length = int(np.sum(attention_mask[0]))
    if length < 1:
        raise ValueError("text must contain at least one token after EOS is appended")

    seq_tokens = tokens[0, :length]
    if seq_tokens.shape[0] > 0 and int(seq_tokens[-1]) == base_infer.eos_id:
        length = length - 1
        seq_tokens = tokens[0, :length]
    if length < 1:
        raise ValueError("text must contain at least one non-EOS token")

    last_token_id = int(seq_tokens[-1])
    prev_token_id = int(seq_tokens[-2]) if length > 1 else -1

    base_reps = base_infer.encode_tokens(tokens, attention_mask)
    reps_last = base_reps[:, length - 2 : length, :] if length > 1 else base_reps[:, length - 1 : length, :]

    policy_dir = _resolve_checkpoint_dir(args.policy_checkpoint)
    policy_config_root = _load_run_config(policy_dir)
    policy_config_raw = policy_config_root.get("policy_model")
    if policy_config_raw is None or isinstance(policy_config_raw, dict) is False:
        raise ValueError("policy checkpoint missing policy_model config")
    training_config = policy_config_root.get("training")
    if training_config is None or isinstance(training_config, dict) is False:
        raise ValueError("policy checkpoint missing training config")
    dtype_name = training_config.get("dtype")
    if isinstance(dtype_name, str) is False:
        raise ValueError("policy training config missing dtype")

    policy_model = _load_policy_model(
        policy_dir,
        dtype=_dtype_from_name(dtype_name),
        policy_config=policy_config_raw,
    )
    if policy_model.config.d_model != reps_last.shape[-1]:
        raise ValueError("policy model d_model must match base model output")

    if reps_last.shape[1] == 1:
        start = policy_model.start_token.astype(reps_last.dtype)
        start = jnp.broadcast_to(start[None, None, :], (reps_last.shape[0], 1, reps_last.shape[2]))
        reps_last = jnp.concatenate([start, reps_last], axis=1)

    logits = policy_model(reps_last, train=False, key=None)
    if logits.shape[0] < 2:
        raise ValueError("policy logits must contain at least two positions")
    logits_last = logits[1].astype(jnp.float32)
    probs = jax.nn.softmax(logits_last, axis=-1)
    probs_np = np.asarray(probs)

    top_k = min(args.top_k, probs_np.shape[0])
    top_indices = np.argsort(-probs_np)[:top_k]

    if prev_token_id >= 0:
        prev_label = repr(base_infer.decode_token(prev_token_id))
    else:
        prev_label = "<start>"
    print("Previous token:", prev_label, f"(id={prev_token_id})")
    print("True last token:", repr(base_infer.decode_token(last_token_id)), f"(id={last_token_id})")
    print("Top predictions:")
    for rank, token_id in enumerate(top_indices, start=1):
        token_text = base_infer.decode_token(int(token_id))
        confidence = float(probs_np[int(token_id)])
        print(f"  {rank}. {repr(token_text)} (id={int(token_id)}): {confidence:.6f}")


if __name__ == "__main__":
    main()
