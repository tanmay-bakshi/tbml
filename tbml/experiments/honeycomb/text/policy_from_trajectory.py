import argparse
import json
import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tbml.experiments.honeycomb.text.inference import TextInference
from tbml.experiments.honeycomb.text.train_recurrent_policy import RecurrentPolicy, RecurrentPolicyConfig
from tbml.experiments.honeycomb.text.train_transformer_policy import (
    PolicyTransformer,
    PolicyTransformerConfig,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Run a policy model over a saved trajectory.")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--policy-checkpoint", type=str, required=True)
    parser.add_argument("--trajectory", type=str, required=True)
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
    policy_type: str,
    policy_config: dict[str, Any],
) -> RecurrentPolicy | PolicyTransformer:
    """Load a policy model from a checkpoint.

    :param checkpoint_dir: Checkpoint directory path.
    :param dtype: Compute dtype to try first.
    :param policy_type: Policy type identifier.
    :param policy_config: Policy configuration dictionary.
    :returns: Deserialized policy model.
    """
    model_path = os.path.join(checkpoint_dir, "model.eqx")
    candidates = [dtype, jnp.float32, jnp.bfloat16, jnp.float16]
    seen: set[jnp.dtype] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        model_key = jax.random.PRNGKey(0)
        if policy_type == "transformer":
            config = PolicyTransformerConfig(**policy_config)
            model = PolicyTransformer(
                config,
                dtype=candidate,
                param_dtype=_param_dtype_for(candidate),
                key=model_key,
            )
        else:
            config = RecurrentPolicyConfig(**policy_config)
            model = RecurrentPolicy(
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


def _format_top_k(
    base: TextInference,
    probs: np.ndarray,
    *,
    top_k: int,
) -> list[tuple[str, int, float]]:
    """Format top-K tokens and probabilities.

    :param base: Base inference helper.
    :param probs: Probability array of shape (V,).
    :param top_k: Number of top tokens to return.
    :returns: List of (token_text, token_id, probability).
    """
    k = min(top_k, probs.shape[0])
    indices = np.argsort(-probs)[:k]
    results: list[tuple[str, int, float]] = []
    for idx in indices:
        token_id = int(idx)
        token_text = base.decode_token(token_id)
        results.append((token_text, token_id, float(probs[token_id])))
    return results


def main() -> None:
    """Run policy logits on a trajectory and print top-k tokens."""
    args = _parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be > 0")

    base = TextInference.from_checkpoint(args.base_checkpoint)
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

    policy_type = policy_config_root.get("policy_type")
    if policy_type is None:
        policy_type = "recurrent_lstm"
    if isinstance(policy_type, str) is False:
        raise ValueError("policy_type must be a string")

    policy_model = _load_policy_model(
        policy_dir,
        dtype=_dtype_from_name(dtype_name),
        policy_type=policy_type,
        policy_config=policy_config_raw,
    )

    trajectory = np.load(args.trajectory)
    if trajectory.ndim != 2:
        raise ValueError("trajectory must have shape (T, D)")
    if trajectory.shape[0] == 0:
        raise ValueError("trajectory must contain at least one token")
    if isinstance(policy_model, RecurrentPolicy):
        expected_dim = policy_model.config.input_dim
    else:
        expected_dim = policy_model.config.d_model
    if trajectory.shape[1] != expected_dim:
        raise ValueError("trajectory dimension must match policy input dimension")

    reps = jnp.asarray(trajectory, dtype=policy_model.dtype)
    reps_batch = reps[None, :, :]
    if isinstance(policy_model, PolicyTransformer):
        attention_mask = jnp.ones((1, reps.shape[0]), dtype=bool)
        logits = policy_model(reps_batch, attention_mask=attention_mask, train=False, key=None)
    else:
        logits = policy_model(reps_batch, train=False, key=None)

    probs = np.asarray(jax.nn.softmax(logits.astype(jnp.float32), axis=-1))

    for idx in range(trajectory.shape[0]):
        header = f"Token {idx + 1}/{trajectory.shape[0]}"
        print(header)
        print("-" * len(header))
        top = _format_top_k(base, probs[0, idx], top_k=args.top_k)
        for rank, (text, token_id, prob) in enumerate(top, start=1):
            print(f"  {rank:>2}. {repr(text)} (id={token_id}): {prob:.6f}")
        print("")

    top_ids = np.argmax(probs[0], axis=-1)
    predicted_tokens = [base.decode_token(int(token_id)) for token_id in top_ids]
    predicted_text = "".join(predicted_tokens)
    print("Predicted text:")
    print(repr(predicted_text))


if __name__ == "__main__":
    main()
