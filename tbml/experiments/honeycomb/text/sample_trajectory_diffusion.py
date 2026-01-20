import argparse
import json
import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tbml.experiments.honeycomb.text.inference import TextInference
from tbml.experiments.honeycomb.text.train_trajectory_diffusion import DiffusionTransformer, DiffusionTransformerConfig


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Sample a trajectory from a diffusion model.")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--diffusion-checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--condition-text", type=str, default="")
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--override-steps", type=int, default=0)
    parser.add_argument("--override-beta-start", type=float, default=0.0)
    parser.add_argument("--override-beta-end", type=float, default=0.0)
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


def _load_diffusion_model(
    checkpoint_dir: str,
    *,
    dtype: jnp.dtype,
    config_raw: dict[str, Any],
) -> DiffusionTransformer:
    """Load a diffusion model from a checkpoint.

    :param checkpoint_dir: Checkpoint directory path.
    :param dtype: Compute dtype to try first.
    :param config_raw: Diffusion model configuration.
    :returns: Deserialized diffusion model.
    """
    config = DiffusionTransformerConfig(**config_raw)
    model_path = os.path.join(checkpoint_dir, "model.eqx")
    candidates = [dtype, jnp.float32, jnp.bfloat16, jnp.float16]
    seen: set[jnp.dtype] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        model_key = jax.random.PRNGKey(0)
        model = DiffusionTransformer(
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
    raise RuntimeError("failed to load diffusion model")


def _linear_noise_schedule(
    *,
    beta_start: float,
    beta_end: float,
    num_steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a linear beta schedule and cumulative alpha products.

    :param beta_start: Starting beta value.
    :param beta_end: Ending beta value.
    :param num_steps: Number of diffusion steps.
    :returns: Tuple of (betas, alpha_cumprod) arrays.
    """
    betas = jnp.linspace(beta_start, beta_end, num_steps, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_cumprod = jnp.cumprod(alphas)
    return betas, alpha_cumprod


def _last_non_eos_index(tokens: np.ndarray, mask: np.ndarray, eos_id: int) -> int:
    """Return the index of the last non-EOS token.

    :param tokens: Token ids of shape (1, T).
    :param mask: Attention mask of shape (1, T).
    :param eos_id: EOS token id.
    :returns: Index of the last non-EOS token.
    """
    length = int(np.sum(mask[0]))
    if length <= 0:
        raise ValueError("no tokens in conditioning text")
    if int(tokens[0, length - 1]) == eos_id:
        length -= 1
    if length <= 0:
        raise ValueError("conditioning text has no non-EOS tokens")
    return length - 1


def _compute_condition(base: TextInference, text: str) -> jnp.ndarray:
    """Compute conditioning embedding from text.

    :param base: Base inference helper.
    :param text: Conditioning text.
    :returns: Conditioning embedding of shape (1, d_model).
    """
    tokens, attention_mask = base.preprocess([text])
    reps = base.encode_tokens(tokens, attention_mask)
    idx = _last_non_eos_index(tokens, attention_mask, base.eos_id)
    return reps[:, idx, :]


def main() -> None:
    """Sample a trajectory and save it to disk."""
    args = _parse_args()
    if args.cfg_scale < 0.0:
        raise ValueError("cfg-scale must be >= 0")
    if args.override_steps < 0:
        raise ValueError("override-steps must be >= 0")
    if args.override_beta_start < 0.0 or args.override_beta_end < 0.0:
        raise ValueError("override beta values must be >= 0")

    base = TextInference.from_checkpoint(args.base_checkpoint)

    diffusion_dir = _resolve_checkpoint_dir(args.diffusion_checkpoint)
    diff_config_root = _load_run_config(diffusion_dir)
    diff_config_raw = diff_config_root.get("diffusion_model")
    if diff_config_raw is None or isinstance(diff_config_raw, dict) is False:
        raise ValueError("diffusion checkpoint missing diffusion_model config")
    training_config = diff_config_root.get("training")
    if training_config is None or isinstance(training_config, dict) is False:
        raise ValueError("diffusion checkpoint missing training config")
    diffusion_cfg = diff_config_root.get("diffusion")
    if diffusion_cfg is None or isinstance(diffusion_cfg, dict) is False:
        raise ValueError("diffusion checkpoint missing diffusion config")

    dtype_name = training_config.get("dtype")
    if isinstance(dtype_name, str) is False:
        raise ValueError("diffusion training config missing dtype")

    num_prefix_tokens = int(diffusion_cfg.get("num_prefix_tokens", 0))
    if num_prefix_tokens <= 0:
        raise ValueError("diffusion config missing num_prefix_tokens")

    num_steps = int(diffusion_cfg.get("num_diffusion_steps", 0))
    beta_start = float(diffusion_cfg.get("beta_start", 0.0))
    beta_end = float(diffusion_cfg.get("beta_end", 0.0))
    if args.override_steps > 0:
        num_steps = args.override_steps
    if args.override_beta_start > 0.0:
        beta_start = args.override_beta_start
    if args.override_beta_end > 0.0:
        beta_end = args.override_beta_end
    if num_steps <= 0:
        raise ValueError("num_diffusion_steps must be > 0")

    diffusion_model = _load_diffusion_model(
        diffusion_dir,
        dtype=_dtype_from_name(dtype_name),
        config_raw=diff_config_raw,
    )

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    x = jax.random.normal(
        init_key,
        (1, num_prefix_tokens, diffusion_model.config.d_model),
        dtype=jnp.float32,
    )

    cond_text = args.condition_text
    use_cond = cond_text != ""
    if args.cfg_scale > 0.0 and use_cond is False:
        raise ValueError("cfg-scale requires a non-empty --condition-text")

    cond_vec: jnp.ndarray | None = None
    if use_cond is True:
        cond_vec = _compute_condition(base, cond_text)

    betas, alpha_cumprod = _linear_noise_schedule(
        beta_start=beta_start,
        beta_end=beta_end,
        num_steps=num_steps,
    )
    alphas = 1.0 - betas

    for t in reversed(range(num_steps)):
        t_batch = jnp.asarray([t], dtype=jnp.int32)
        alpha_t = alphas[t].astype(jnp.float32)
        alpha_bar_t = alpha_cumprod[t].astype(jnp.float32)

        if use_cond is True and args.cfg_scale > 0.0:
            if cond_vec is None:
                raise ValueError("conditioning vector missing")
            v_cond = diffusion_model(x, t_batch, cond_vec, train=False, key=None)
            v_uncond = diffusion_model(
                x,
                t_batch,
                diffusion_model.null_cond[None, :],
                train=False,
                key=None,
            )
            v_pred = v_uncond + args.cfg_scale * (v_cond - v_uncond)
        else:
            if use_cond is True:
                if cond_vec is None:
                    raise ValueError("conditioning vector missing")
                cond = cond_vec
            else:
                cond = diffusion_model.null_cond[None, :]
            v_pred = diffusion_model(x, t_batch, cond, train=False, key=None)

        v_pred = v_pred.astype(jnp.float32)
        sqrt_alpha = jnp.sqrt(alpha_t)
        sqrt_one_minus = jnp.sqrt(1.0 - alpha_bar_t)
        eps = sqrt_one_minus * x + sqrt_alpha * v_pred
        pred_mean = (x - (1.0 - alpha_t) / sqrt_one_minus * eps) / sqrt_alpha

        if t > 0:
            alpha_bar_prev = alpha_cumprod[t - 1].astype(jnp.float32)
            beta_t = betas[t].astype(jnp.float32)
            variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, x.shape, dtype=jnp.float32)
            x = pred_mean + jnp.sqrt(variance) * noise
        else:
            x = pred_mean

    trajectory = np.asarray(x[0])
    np.save(args.output, trajectory)
    print("Saved trajectory:", trajectory.shape)


if __name__ == "__main__":
    main()
