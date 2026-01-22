import argparse
import math
import json
import os
import queue
import threading
import time
from datetime import datetime
import traceback
from typing import Iterable, Iterator, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding
from tensorboardX import SummaryWriter  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from tbml.experiments.honeycomb.loss import sigreg_loss_views
from tbml.experiments.honeycomb.text.dataset import (
    MMapTokenDataset,
    _build_tokenizer,
    iter_text_batches,
)
from tbml.experiments.honeycomb.text.model import TextTransformer, TextTransformerConfig
from tbml.nn import RMSNorm
from tbml.optimizers import MuonWithAdamWFallback, MuonWithAdamWFallbackState, build_muon_masks


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train a text encoder with LeJEPA.")
    parser.add_argument("--runs-folder", type=str, required=True)
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup-steps", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=32)
    parser.add_argument("--num-devices", type=int, default=0)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--preview-views", action="store_true")

    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--predictor-n-layers", type=int, default=None)
    parser.add_argument("--decoder-n-layers", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--resid-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--pope-base", type=float, default=10000.0)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--attn-type", type=str, default="pope", choices=["pope", "rope"])
    parser.add_argument("--causal-attention", type=str, default="true", choices=["true", "false"])

    parser.add_argument("--num-global-views", type=int, default=2)
    parser.add_argument("--num-local-views", type=int, default=6)
    parser.add_argument("--global-mask-min", type=float, default=None)
    parser.add_argument("--global-mask-max", type=float, default=None)
    parser.add_argument("--local-mask-min", type=float, default=None)
    parser.add_argument("--local-mask-max", type=float, default=None)

    parser.add_argument("--sigreg-weight", type=float, default=0.25)
    parser.add_argument("--sigreg-slices", type=int, default=256)
    parser.add_argument("--sigreg-seed", type=int, default=0)
    parser.add_argument("--pred-loss", type=str, default="mse", choices=["mse", "cosine"])
    parser.add_argument("--seq-loss-weight", type=float, default=0.5)
    parser.add_argument("--span-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoder-loss-weight", type=float, default=0.5)
    parser.add_argument("--encoder-mlm-weight", type=float, default=0.0)
    parser.add_argument("--span-tokenwise", action="store_true")
    parser.add_argument("--mask-token-input", action="store_true")

    parser.add_argument("--muon-lr", type=float, default=1e-3)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-nesterov", dest="muon_nesterov", action="store_true")
    parser.add_argument("--no-muon-nesterov", dest="muon_nesterov", action="store_false")
    parser.add_argument("--muon-n-steps", type=int, default=5)
    parser.add_argument("--muon-ns-eps", type=float, default=1e-7)

    parser.add_argument("--adamw-lr", type=float, default=1e-3)
    parser.add_argument("--adamw-betas", type=str, default="0.9,0.999")
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.01)

    parser.set_defaults(muon_nesterov=True)
    return parser.parse_args()


def _parse_betas(value: str) -> tuple[float, float]:
    """Parse a comma-separated float pair.

    :param value: Input string of the form "a,b".
    :returns: Parsed float pair.
    """
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("expected two comma-separated values for betas")
    return float(parts[0]), float(parts[1])


def _parse_bool(value: str) -> bool:
    """Parse a boolean string.

    :param value: Input string ("true" or "false").
    :returns: Parsed boolean.
    """
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError("expected 'true' or 'false'")


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


def _build_run_dir(runs_folder: str) -> str:
    """Create a unique run directory.

    :param runs_folder: Root directory for runs.
    :returns: Newly created run directory path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_folder, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def _decode_tokens(
    tokenizer: "Tokenizer",
    token_ids: np.ndarray,
    *,
    pad_id: int,
) -> str:
    """Decode a token sequence, trimming padding.

    :param tokenizer: Tokenizer instance.
    :param token_ids: Token ids of shape (T,).
    :param pad_id: Padding token id.
    :returns: Decoded string.
    """
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be 1D")
    pad_positions = np.where(token_ids == pad_id)[0]
    if pad_positions.shape[0] > 0:
        limit = int(pad_positions[0])
        trimmed = token_ids[:limit]
    else:
        trimmed = token_ids
    return tokenizer.decode(trimmed.tolist(), skip_special_tokens=False)


def _flatten_param_names(params: eqx.Module) -> list[str]:
    """Flatten parameter PyTree paths into dotted names.

    :param params: Parameter PyTree.
    :returns: List of flattened parameter names.
    """
    paths_and_values, _ = jax.tree_util.tree_flatten_with_path(params)
    names: list[str] = []
    for path, _value in paths_and_values:
        parts: list[str] = []
        for key in path:
            if isinstance(key, jax.tree_util.GetAttrKey):
                parts.append(key.name)
            elif isinstance(key, jax.tree_util.DictKey):
                parts.append(str(key.key))
            elif isinstance(key, jax.tree_util.SequenceKey):
                parts.append(str(key.idx))
            else:
                parts.append(str(key))
        names.append(".".join(parts))
    return names


T = TypeVar("T")


def _replicate_tree(tree: T, num_devices: int) -> T:
    """Replicate array leaves along a leading device axis.

    :param tree: PyTree of parameters or optimizer state.
    :param num_devices: Number of devices to replicate across.
    :returns: Replicated PyTree with a leading device axis.
    """
    return cast(
        T,
        jax.tree_util.tree_map(
            lambda value: jnp.broadcast_to(value, (num_devices,) + value.shape)
            if eqx.is_array(value)
            else value,
            tree,
        ),
    )


def _unreplicate(tree: T) -> T:
    """Select the first replica from a replicated PyTree.

    :param tree: Replicated PyTree.
    :returns: Unreplicated PyTree.
    """
    return cast(
        T,
        jax.tree_util.tree_map(
            lambda value: value[0] if isinstance(value, jax.Array) else value,
            tree,
        ),
    )


def _to_host(tree: T) -> T:
    """Move a PyTree of arrays to host memory.

    :param tree: PyTree containing JAX arrays.
    :returns: PyTree with host arrays.
    """
    return cast(T, jax.device_get(tree))


def _block_until_ready(tree: T) -> None:
    """Block until all arrays in a PyTree are ready.

    :param tree: PyTree containing JAX arrays.
    """

    def _block(value: object) -> None:
        if isinstance(value, jax.Array):
            value.block_until_ready()
        return None

    jax.tree_util.tree_map(_block, tree)


def _add_trees(tree_a: T, tree_b: T) -> T:
    """Add two PyTrees with optional None leaves.

    :param tree_a: First PyTree.
    :param tree_b: Second PyTree.
    :returns: PyTree containing summed leaves.
    """

    def _add(a: object, b: object) -> object:
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return a + b

    return cast(T, jax.tree_util.tree_map(_add, tree_a, tree_b))


def _scale_tree(tree: T, scale: int) -> T:
    """Scale all array leaves in a PyTree by a scalar.

    :param tree: PyTree containing arrays.
    :param scale: Scalar divisor.
    :returns: PyTree with scaled array leaves.
    """
    scale_f = jnp.asarray(scale, dtype=jnp.float32)
    return cast(
        T,
        jax.tree_util.tree_map(
            lambda value: value / scale_f if value is not None else None,
            tree,
        ),
    )


def _cast_tree_dtype(tree: T, dtype: jnp.dtype) -> T:
    """Cast array leaves in a PyTree to a dtype.

    :param tree: PyTree containing array leaves.
    :param dtype: Target dtype for array leaves.
    :returns: PyTree with casted array leaves.
    """
    return cast(
        T,
        jax.tree_util.tree_map(
            lambda value: value.astype(dtype) if eqx.is_array(value) else value,
            tree,
        ),
    )


def _build_sharding(devices: list[jax.Device]) -> tuple[Mesh, NamedSharding, NamedSharding]:
    """Build sharding helpers for data and replicated params.

    :param devices: Devices to shard across.
    :returns: Tuple of (mesh, data_sharding, replicated_sharding).
    """
    mesh = Mesh(np.array(devices), ("data",))
    data_sharding = NamedSharding(mesh, PartitionSpec("data"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    return mesh, data_sharding, replicated_sharding


def _devices_for_platform(platform: str) -> list[jax.Device]:
    """List JAX devices matching a platform name.

    :param platform: Platform string such as "cpu" or "gpu".
    :returns: List of matching devices.
    """
    return [device for device in jax.devices() if device.platform == platform]


def _prefetch_to_device(
    iterator: Iterable[np.ndarray],
    size: int,
    sharding: Sharding,
) -> Iterator[Array]:
    """Prefetch batches to devices using a background thread.

    :param iterator: Host iterator yielding sharded host batches.
    :param size: Prefetch depth.
    :param sharding: Sharding specification for device placement.
    :returns: Iterator yielding device-resident batches.
    """
    if size <= 0:
        raise ValueError("prefetch size must be > 0")

    work_queue: queue.Queue[object] = queue.Queue(maxsize=size)
    sentinel = object()

    def _worker() -> None:
        """Prefetch batches from the host iterator onto devices."""
        try:
            for item in iterator:
                device_item = jax.device_put(item, device=sharding)
                work_queue.put(device_item)
        except Exception as exc:
            stack = traceback.format_exc()
            work_queue.put(RuntimeError(f"prefetch worker failed: {exc}\n{stack}"))
        finally:
            work_queue.put(sentinel)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while True:
        payload = work_queue.get()
        if payload is sentinel:
            break
        if isinstance(payload, Exception):
            raise payload
        yield cast(Array, payload)


def _iter_batches(
    dataset: MMapTokenDataset,
    *,
    batch_size: int,
    max_seq_len: int,
    shuffle_buffer: int,
    seed: int,
    num_devices: int,
    per_device_batch: int,
) -> Iterable[np.ndarray]:
    """Yield device-sharded batches from the dataset.

    :param dataset: Dataset instance.
    :param batch_size: Global batch size.
    :param max_seq_len: Maximum sequence length.
    :param shuffle_buffer: Shuffle buffer for sample order.
    :param seed: Random seed for shuffle order.
    :param num_devices: Number of devices.
    :param per_device_batch: Batch size per device.
    :returns: Iterable of sharded token batches.
    """
    host_iter = iter_text_batches(
        dataset,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )
    for tokens in host_iter:
        tokens = tokens.reshape((num_devices, per_device_batch, max_seq_len))
        yield tokens


def _mask_spans(
    tokens: Array,
    key: Array,
    *,
    min_ratio: float,
    max_ratio: float,
    pad_id: int,
    eos_id: int,
    mask_id: int,
    use_mask_token: bool,
    poisson_lambda: float = 3.0,
) -> tuple[Array, Array, Array]:
    """Apply span masking to tokens using BART-style text infilling.

    Masked tokens are excluded from attention but the sequence length is unchanged.

    :param tokens: Token ids of shape (B, T).
    :param key: PRNG key.
    :param min_ratio: Minimum masking ratio.
    :param max_ratio: Maximum masking ratio.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :param mask_id: Mask token id.
    :param use_mask_token: Whether to replace masked positions with mask tokens.
    :param poisson_lambda: Poisson lambda for span lengths.
    :returns: Tuple of (token ids, mask positions, attention mask).
    """
    if min_ratio < 0.0 or min_ratio > 1.0:
        raise ValueError("min_ratio must be in [0, 1]")
    if max_ratio < 0.0 or max_ratio > 1.0:
        raise ValueError("max_ratio must be in [0, 1]")
    if min_ratio > max_ratio:
        raise ValueError("min_ratio must be <= max_ratio")
    if poisson_lambda <= 0.0:
        raise ValueError("poisson_lambda must be > 0")

    key_ratio, key_spans = jax.random.split(key)
    ratios = jax.random.uniform(
        key_ratio,
        shape=(tokens.shape[0],),
        minval=min_ratio,
        maxval=max_ratio,
    )
    keys = jax.random.split(key_spans, tokens.shape[0])
    seq_len = tokens.shape[1]
    idx = jnp.arange(seq_len, dtype=jnp.int32)
    idx_row = idx[None, :]
    all_lengths = jnp.arange(1, seq_len + 1, dtype=jnp.int32)[:, None]

    def _one_sample(
        sample_tokens: Array,
        ratio: Array,
        sample_key: Array,
    ) -> tuple[Array, Array, Array]:
        """Mask spans for a single sequence.

        :param sample_tokens: Token ids of shape (T,).
        :param ratio: Masking ratio scalar.
        :param sample_key: PRNG key.
        :returns: Tuple of (masked tokens, mask positions, attention mask).
        """
        key_len, key_start = jax.random.split(sample_key)
        eligible = jnp.logical_and(sample_tokens != pad_id, sample_tokens != eos_id)
        valid_len = jnp.sum(eligible).astype(jnp.int32)
        target = jnp.floor(ratio * valid_len).astype(jnp.int32)
        target = jnp.minimum(target, jnp.maximum(valid_len - 1, 0))
        span_lengths = jax.random.poisson(key_len, poisson_lambda, shape=(seq_len,))
        span_lengths = jnp.maximum(span_lengths, 1)
        start_scores = jax.random.uniform(key_start, shape=(seq_len, seq_len), minval=0.0, maxval=1.0)

        def _body(i: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
            mask, masked_count = state
            remaining = target - masked_count
            use = remaining > 0
            span_len = jnp.minimum(span_lengths[i], remaining)
            span_len = jnp.maximum(span_len, 1)
            valid = jnp.logical_and(eligible, jnp.logical_not(mask))
            valid_int = valid.astype(jnp.int32)
            prefix = jnp.concatenate(
                [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(valid_int)]
            )
            end = idx_row + all_lengths
            valid_window = end <= seq_len
            prefix_end = jnp.take(prefix, end, mode="clip")
            prefix_start = prefix[idx_row]
            window_sum = jnp.where(valid_window, prefix_end - prefix_start, 0)
            length_index = jnp.maximum(span_len - 1, 0)
            possible = jnp.take(window_sum, length_index, axis=0) == span_len
            any_possible = jnp.any(possible)
            scores = jnp.where(possible, start_scores[i], -jnp.inf)
            start = jnp.argmax(scores)
            span_positions = jnp.logical_and(idx >= start, idx < start + span_len)
            apply = jnp.logical_and(use, any_possible)
            new_mask = jnp.where(apply, jnp.logical_or(mask, span_positions), mask)
            new_count = jnp.where(apply, masked_count + span_len, masked_count)
            return new_mask, new_count

        init_mask = jnp.zeros((seq_len,), dtype=jnp.bool_)
        init_count = jnp.asarray(0, dtype=jnp.int32)
        final_mask, _final_count = jax.lax.fori_loop(
            0, seq_len, _body, (init_mask, init_count)
        )
        if use_mask_token is True:
            attention_mask = eligible
        else:
            attention_mask = jnp.logical_and(eligible, jnp.logical_not(final_mask))

        def _with_fallback(_: None) -> tuple[Array, Array]:
            fallback_idx = jnp.argmax(eligible)
            attn = attention_mask.at[fallback_idx].set(True)
            mask = final_mask.at[fallback_idx].set(False)
            return attn, mask

        def _no_fallback(_: None) -> tuple[Array, Array]:
            return attention_mask, final_mask

        attention_mask, final_mask = jax.lax.cond(
            jnp.any(attention_mask),
            _no_fallback,
            _with_fallback,
            operand=None,
        )
        if use_mask_token is True:
            masked_tokens = jnp.where(final_mask, jnp.asarray(mask_id, dtype=sample_tokens.dtype), sample_tokens)
        else:
            masked_tokens = sample_tokens
        return masked_tokens, final_mask, attention_mask

    masked, positions, attn_mask = jax.vmap(_one_sample)(tokens, ratios, keys)
    return masked, positions, attn_mask


def _mask_views(
    tokens: Array,
    key: Array,
    *,
    num_views: int,
    min_ratio: float,
    max_ratio: float,
    pad_id: int,
    eos_id: int,
    mask_id: int,
    use_mask_token: bool,
) -> tuple[Array, Array, Array]:
    """Generate multiple masked views of the token batch.

    :param tokens: Token ids of shape (B, T).
    :param key: PRNG key.
    :param num_views: Number of views to generate.
    :param min_ratio: Minimum masking ratio.
    :param max_ratio: Maximum masking ratio.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :param mask_id: Mask token id.
    :param use_mask_token: Whether to replace masked positions with mask tokens.
    :returns: Tuple of (masked views, mask positions, attention masks).
    """
    if num_views == 0:
        empty_views = jnp.zeros((tokens.shape[0], 0, tokens.shape[1]), dtype=tokens.dtype)
        empty_mask = jnp.zeros((tokens.shape[0], 0, tokens.shape[1]), dtype=jnp.bool_)
        empty_attn = jnp.zeros((tokens.shape[0], 0, tokens.shape[1]), dtype=jnp.bool_)
        return empty_views, empty_mask, empty_attn

    keys = jax.random.split(key, num_views)

    def _one_view(view_key: Array) -> tuple[Array, Array, Array]:
        """Mask tokens using the provided view key.

        :param view_key: PRNG key for masking.
        :returns: Tuple of (masked token ids, mask positions, attention mask).
        """
        return _mask_spans(
            tokens,
            view_key,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            pad_id=pad_id,
            eos_id=eos_id,
            mask_id=mask_id,
            use_mask_token=use_mask_token,
        )

    views, masks, attn_masks = jax.vmap(_one_view)(keys)
    return (
        jnp.transpose(views, (1, 0, 2)),
        jnp.transpose(masks, (1, 0, 2)),
        jnp.transpose(attn_masks, (1, 0, 2)),
    )


def _build_views(
    tokens: Array,
    key: Array,
    *,
    num_global_views: int,
    num_local_views: int,
    global_mask_min: float,
    global_mask_max: float,
    local_mask_min: float,
    local_mask_max: float,
    pad_id: int,
    eos_id: int,
    mask_id: int,
    use_mask_token: bool,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Build masked views for global and local crops.

    :param tokens: Token ids of shape (B, T).
    :param key: PRNG key.
    :param num_global_views: Number of global views.
    :param num_local_views: Number of local views.
    :param global_mask_min: Minimum global masking ratio.
    :param global_mask_max: Maximum global masking ratio.
    :param local_mask_min: Minimum local masking ratio.
    :param local_mask_max: Maximum local masking ratio.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :param mask_id: Mask token id.
    :param use_mask_token: Whether to replace masked positions with mask tokens.
    :returns: Tuple of (model key, global views, global masks, global attn, local views, local masks, local attn).
    """
    model_key, global_key, local_key = jax.random.split(key, 3)
    global_views, global_masks, global_attn = _mask_views(
        tokens,
        global_key,
        num_views=num_global_views,
        min_ratio=global_mask_min,
        max_ratio=global_mask_max,
        pad_id=pad_id,
        eos_id=eos_id,
        mask_id=mask_id,
        use_mask_token=use_mask_token,
    )
    local_views, local_masks, local_attn = _mask_views(
        tokens,
        local_key,
        num_views=num_local_views,
        min_ratio=local_mask_min,
        max_ratio=local_mask_max,
        pad_id=pad_id,
        eos_id=eos_id,
        mask_id=mask_id,
        use_mask_token=use_mask_token,
    )
    return (
        model_key,
        global_views,
        global_masks,
        global_attn,
        local_views,
        local_masks,
        local_attn,
    )


def _combine_views(
    global_views: Array,
    global_masks: Array,
    global_attn: Array,
    local_views: Array,
    local_masks: Array,
    local_attn: Array,
) -> tuple[Array, Array, Array]:
    """Combine global and local views into a single view batch.

    :param global_views: Global view tokens.
    :param global_masks: Global view mask positions.
    :param global_attn: Global view attention masks.
    :param local_views: Local view tokens.
    :param local_masks: Local view mask positions.
    :param local_attn: Local view attention masks.
    :returns: Tuple of (views, mask positions, attention masks).
    """
    views = jnp.concatenate([global_views, local_views], axis=1)
    mask_positions = jnp.concatenate([global_masks, local_masks], axis=1)
    view_attn = jnp.concatenate([global_attn, local_attn], axis=1)
    return views, mask_positions, view_attn


def _pool_last(
    reps: Array,
    attention_mask: Array,
) -> Array:
    """Pool the last valid token representation.

    :param reps: Token representations of shape (B, T, D).
    :param attention_mask: Attention mask of shape (B, T).
    :returns: Pooled representations of shape (B, D).
    """
    if reps.ndim != 3:
        raise ValueError("reps must have shape (B, T, D)")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape (B, T)")
    if reps.shape[:2] != attention_mask.shape:
        raise ValueError("reps and attention_mask must align on (B, T)")

    mask = attention_mask.astype(bool)
    positions = jnp.arange(reps.shape[1], dtype=jnp.int32)
    positions = jnp.broadcast_to(positions[None, :], mask.shape)
    masked_positions = jnp.where(mask, positions, jnp.full_like(positions, -1))
    last_idx = jnp.max(masked_positions, axis=1)
    last_idx = jnp.maximum(last_idx, 0)
    idx = last_idx[:, None, None]
    idx = jnp.broadcast_to(idx, (idx.shape[0], 1, reps.shape[-1]))
    return jnp.take_along_axis(reps, idx, axis=1).squeeze(axis=1)


def _encode_views(
    model: TextTransformer,
    tokens: Array,
    *,
    train: bool,
    key: Array,
    num_global_views: int,
    num_local_views: int,
    global_mask_min: float,
    global_mask_max: float,
    local_mask_min: float,
    local_mask_max: float,
    pad_id: int,
    eos_id: int,
    mask_id: int,
    use_mask_token: bool,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Encode masked views into pooled embeddings and token representations.

    :param model: Text transformer encoder.
    :param tokens: Token ids of shape (B, T).
    :param train: Whether to enable dropout.
    :param key: PRNG key.
    :param num_global_views: Number of global views.
    :param num_local_views: Number of local views.
    :param global_mask_min: Minimum global masking ratio.
    :param global_mask_max: Maximum global masking ratio.
    :param local_mask_min: Minimum local masking ratio.
    :param local_mask_max: Maximum local masking ratio.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :param mask_id: Mask token id.
    :param use_mask_token: Whether to replace masked positions with mask tokens.
    :returns: Tuple of (pooled_post, pooled_pre, token_post, token_pre, mask_positions, view_attn).
    """
    if num_global_views + num_local_views <= 0:
        raise ValueError("at least one view must be requested")

    tokens = jnp.where(tokens == eos_id, jnp.asarray(pad_id, dtype=tokens.dtype), tokens)

    (
        model_key,
        global_views,
        global_masks,
        global_attn,
        local_views,
        local_masks,
        local_attn,
    ) = _build_views(
        tokens,
        key,
        num_global_views=num_global_views,
        num_local_views=num_local_views,
        global_mask_min=global_mask_min,
        global_mask_max=global_mask_max,
        local_mask_min=local_mask_min,
        local_mask_max=local_mask_max,
        pad_id=pad_id,
        eos_id=eos_id,
        mask_id=mask_id,
        use_mask_token=use_mask_token,
    )
    views, _mask_positions, view_attn = _combine_views(
        global_views,
        global_masks,
        global_attn,
        local_views,
        local_masks,
        local_attn,
    )

    sample_view = tokens[:, None, :]
    sample_mask = jnp.zeros_like(sample_view, dtype=jnp.bool_)
    sample_attn = tokens != pad_id
    sample_attn = sample_attn[:, None, :]

    views = jnp.concatenate([views, sample_view], axis=1)
    mask_positions = jnp.concatenate([_mask_positions, sample_mask], axis=1)
    view_attn = jnp.concatenate([view_attn, sample_attn], axis=1)

    bsz, num_views, seq_len = views.shape
    flat_views = views.reshape((bsz * num_views, seq_len))
    flat_mask = view_attn.reshape((bsz * num_views, seq_len))
    if use_mask_token is True:
        pool_mask = jnp.logical_and(view_attn, jnp.logical_not(mask_positions))
    else:
        pool_mask = view_attn
    flat_pool_mask = pool_mask.reshape((bsz * num_views, seq_len))
    reps_pre, reps_post, _pooled_post = model.forward_with_intermediates(
        flat_views,
        flat_mask,
        train=train,
        key=model_key,
    )
    pooled_pre = _pool_last(reps_pre, flat_pool_mask)
    pooled_post = _pool_last(reps_post, flat_pool_mask)
    token_pre = reps_pre.reshape((bsz, num_views, seq_len, reps_pre.shape[-1]))
    token_post = reps_post.reshape((bsz, num_views, seq_len, reps_post.shape[-1]))
    pooled_post = pooled_post.reshape((bsz, num_views, pooled_post.shape[-1]))
    pooled_pre = pooled_pre.reshape((bsz, num_views, pooled_pre.shape[-1]))
    return pooled_post, pooled_pre, token_post, token_pre, mask_positions, view_attn


def _span_ids(mask: Array) -> Array:
    """Assign a span id to each masked position.

    :param mask: Boolean mask of shape (T,).
    :returns: Integer span ids of shape (T,), 0 for unmasked positions.
    """
    if mask.ndim != 1:
        raise ValueError("mask must be 1D")
    mask_int = mask.astype(jnp.int32)
    prev = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), mask_int[:-1]])
    start = jnp.logical_and(mask_int == 1, prev == 0)
    span_ids = jnp.cumsum(start.astype(jnp.int32)) * mask_int
    return span_ids


def _predictor_span_loss(
    sample_reps: Array,
    view_reps: Array,
    mask_positions: Array,
    *,
    base_norm: RMSNorm,
    predictor_norm: RMSNorm,
) -> Array:
    """Compute predictor span reconstruction loss for a batch of views.

    :param sample_reps: Sample token reps of shape (B, T, D) pre-base-norm.
    :param view_reps: Predictor reps of shape (B, V, T, D) pre-predictor-norm.
    :param mask_positions: Mask positions of shape (B, V, T).
    :param base_norm: Base model RMSNorm module.
    :param predictor_norm: Predictor RMSNorm module.
    :returns: Scalar loss.
    """
    if sample_reps.ndim != 3:
        raise ValueError("sample_reps must have shape (B, T, D)")
    if view_reps.ndim != 4:
        raise ValueError("view_reps must have shape (B, V, T, D)")
    if mask_positions.ndim != 3:
        raise ValueError("mask_positions must have shape (B, V, T)")
    if view_reps.shape[:3] != mask_positions.shape:
        raise ValueError("view_reps and mask_positions must align on (B, V, T)")
    if sample_reps.shape[0] != view_reps.shape[0]:
        raise ValueError("batch sizes must match")
    if sample_reps.shape[1] != view_reps.shape[2]:
        raise ValueError("sequence lengths must match")

    num_segments = sample_reps.shape[1] + 1

    def _loss_for_view(
        sample_rep: Array,
        view_rep: Array,
        mask: Array,
    ) -> Array:
        span_ids = _span_ids(mask)
        sum_sample = jax.ops.segment_sum(sample_rep, span_ids, num_segments)
        sum_view = jax.ops.segment_sum(view_rep, span_ids, num_segments)
        counts = jax.ops.segment_sum(mask.astype(jnp.float32), span_ids, num_segments)
        denom = jnp.maximum(counts[:, None], 1.0)
        mean_sample = sum_sample / denom
        mean_view = sum_view / denom
        mean_sample = base_norm(mean_sample).astype(jnp.float32)
        mean_view = predictor_norm(mean_view).astype(jnp.float32)
        mse = jnp.mean(jnp.square(mean_sample - mean_view), axis=-1)
        valid = counts > 0
        valid_count = jnp.sum(valid)
        return jnp.where(valid_count > 0, jnp.sum(mse * valid) / valid_count, 0.0)

    def _loss_for_sample(
        sample_rep: Array,
        view_rep: Array,
        mask: Array,
    ) -> Array:
        view_losses = jax.vmap(_loss_for_view, in_axes=(None, 0, 0))(sample_rep, view_rep, mask)
        return jnp.mean(view_losses)

    losses = jax.vmap(_loss_for_sample, in_axes=(0, 0, 0))(sample_reps, view_reps, mask_positions)
    return jnp.mean(losses)


def _predictor_span_loss_tokenwise(
    sample_reps: Array,
    view_reps: Array,
    mask_positions: Array,
    *,
    base_norm: RMSNorm,
    predictor_norm: RMSNorm,
) -> Array:
    """Compute predictor span loss using per-token reconstruction.

    :param sample_reps: Sample token reps of shape (B, T, D) pre-base-norm.
    :param view_reps: Predictor reps of shape (B, V, T, D) pre-predictor-norm.
    :param mask_positions: Mask positions of shape (B, V, T).
    :param base_norm: Base model RMSNorm module.
    :param predictor_norm: Predictor RMSNorm module.
    :returns: Scalar loss.
    """
    if sample_reps.ndim != 3:
        raise ValueError("sample_reps must have shape (B, T, D)")
    if view_reps.ndim != 4:
        raise ValueError("view_reps must have shape (B, V, T, D)")
    if mask_positions.ndim != 3:
        raise ValueError("mask_positions must have shape (B, V, T)")
    if view_reps.shape[:3] != mask_positions.shape:
        raise ValueError("view_reps and mask_positions must align on (B, V, T)")
    if sample_reps.shape[0] != view_reps.shape[0]:
        raise ValueError("batch sizes must match")
    if sample_reps.shape[1] != view_reps.shape[2]:
        raise ValueError("sequence lengths must match")

    num_segments = sample_reps.shape[1] + 1

    def _loss_for_view(sample_rep: Array, view_rep: Array, mask: Array) -> Array:
        span_ids = _span_ids(mask)
        sample_post = base_norm(sample_rep).astype(jnp.float32)
        sample_post = jax.lax.stop_gradient(sample_post)
        view_post = predictor_norm(view_rep).astype(jnp.float32)
        mse = jnp.mean(jnp.square(sample_post - view_post), axis=-1)
        mask_f = mask.astype(jnp.float32)
        span_loss = jax.ops.segment_sum(mse * mask_f, span_ids, num_segments)
        counts = jax.ops.segment_sum(mask_f, span_ids, num_segments)
        denom = jnp.maximum(counts, 1.0)
        mean_span = span_loss / denom
        valid = counts > 0
        valid_count = jnp.sum(valid)
        return jnp.where(valid_count > 0, jnp.sum(mean_span * valid) / valid_count, 0.0)

    def _loss_for_sample(sample_rep: Array, view_rep: Array, mask: Array) -> Array:
        view_losses = jax.vmap(_loss_for_view, in_axes=(None, 0, 0))(sample_rep, view_rep, mask)
        return jnp.mean(view_losses)

    losses = jax.vmap(_loss_for_sample, in_axes=(0, 0, 0))(sample_reps, view_reps, mask_positions)
    return jnp.mean(losses)


def _decoder_loss(
    model: TextTransformer,
    sample_tokens: Array,
    view_pred_reps: Array,
    view_masks: Array,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    key: Array,
) -> Array:
    """Compute decoder autoregressive loss over sampled masked spans.

    :param model: TextTransformer containing the decoder and embeddings.
    :param sample_tokens: Token ids of shape (B, T) with EOS removed.
    :param view_pred_reps: Predictor outputs of shape (B, V, T, D) pre-predictor-norm.
    :param view_masks: Mask positions of shape (B, V, T) for the decoder views.
    :param bos_id: BOS token id.
    :param eos_id: EOS token id.
    :param pad_id: Padding token id.
    :param key: PRNG key for decoder dropout.
    :returns: Scalar decoder loss.
    """
    if sample_tokens.ndim != 2:
        raise ValueError("sample_tokens must have shape (B, T)")
    if view_pred_reps.ndim != 4:
        raise ValueError("view_pred_reps must have shape (B, V, T, D)")
    if view_masks.ndim != 3:
        raise ValueError("view_masks must have shape (B, V, T)")
    if view_pred_reps.shape[:3] != view_masks.shape:
        raise ValueError("view_pred_reps and view_masks must align on (B, V, T)")
    if sample_tokens.shape[0] != view_pred_reps.shape[0]:
        raise ValueError("batch sizes must match")
    if sample_tokens.shape[1] != view_pred_reps.shape[2]:
        raise ValueError("sequence lengths must match")
    view_pred_reps = model.predictor.final_norm(view_pred_reps)
    bsz, num_views, seq_len, dim = view_pred_reps.shape
    flat_masks = view_masks.reshape((bsz * num_views, seq_len))
    flat_pred = view_pred_reps.reshape((bsz * num_views, seq_len, dim))
    sample_tokens = sample_tokens.astype(jnp.int32)
    sample_tokens_flat = jnp.repeat(sample_tokens, repeats=num_views, axis=0)

    positions = jnp.arange(seq_len, dtype=jnp.int32)
    flat_keys = jax.random.split(key, flat_masks.shape[0])

    def _pick_span(mask: Array, span_key: Array) -> tuple[Array, Array, Array]:
        span_ids = _span_ids(mask)
        pos_for_min = jnp.where(span_ids > 0, positions, jnp.full_like(positions, seq_len))
        pos_for_max = jnp.where(span_ids > 0, positions, jnp.full_like(positions, -1))
        start = jax.ops.segment_min(pos_for_min, span_ids, seq_len + 1)
        end = jax.ops.segment_max(pos_for_max, span_ids, seq_len + 1)
        length = jnp.where(end >= start, end - start + 1, 0)
        length = length.at[0].set(0)
        valid = length > 0
        valid = valid.at[0].set(False)
        count = jnp.sum(valid)

        def _choose() -> tuple[Array, Array, Array]:
            probs = valid.astype(jnp.float32)
            probs = probs / jnp.sum(probs)
            idx = jax.random.choice(span_key, valid.shape[0], p=probs)
            return start[idx], length[idx], jnp.asarray(True)

        def _fallback() -> tuple[Array, Array, Array]:
            return jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)

        return jax.lax.cond(count > 0, _choose, _fallback)

    span_starts, span_lengths, span_valid = jax.vmap(_pick_span)(flat_masks, flat_keys)
    offsets = jnp.arange(seq_len, dtype=jnp.int32)
    offsets_with_bos = jnp.arange(seq_len + 1, dtype=jnp.int32)
    chunk_key = jax.random.split(key, 1)[0]
    tokens_view = sample_tokens_flat
    pred_view = flat_pred
    idxs = span_starts[:, None] + offsets[None, :]
    idxs = jnp.clip(idxs, 0, seq_len - 1)
    span_tokens = jnp.take_along_axis(tokens_view, idxs, axis=1)
    span_mem = jnp.take_along_axis(pred_view, idxs[:, :, None], axis=1)

    span_mask = offsets[None, :] < span_lengths[:, None]
    span_tokens = jnp.where(span_mask, span_tokens, pad_id)
    span_mem = jnp.where(span_mask[:, :, None], span_mem, jnp.zeros_like(span_mem))

    bos_col = jnp.full((span_tokens.shape[0], 1), bos_id, dtype=span_tokens.dtype)
    dec_in = jnp.concatenate([bos_col, span_tokens], axis=1)
    dec_mask = offsets_with_bos[None, :] < (span_lengths + 1)[:, None]
    dec_mask = jnp.logical_and(dec_mask, span_valid[:, None])

    dec_targets = jnp.full((span_tokens.shape[0], seq_len + 1), pad_id, dtype=span_tokens.dtype)
    dec_targets = dec_targets.at[:, :seq_len].set(span_tokens)
    eos_pos = jnp.clip(span_lengths, 0, seq_len)
    row_idx = jnp.arange(span_tokens.shape[0], dtype=jnp.int32)
    dec_targets = dec_targets.at[row_idx, eos_pos].set(eos_id)

    target_mask = offsets_with_bos[None, :] < (span_lengths + 1)[:, None]
    target_mask = jnp.logical_and(target_mask, span_valid[:, None])

    mem_mask = jnp.logical_and(span_mask, span_valid[:, None])
    any_mem = jnp.any(mem_mask, axis=1)
    fallback = jnp.zeros_like(mem_mask)
    fallback = fallback.at[:, 0].set(True)
    mem_mask = jnp.where(any_mem[:, None], mem_mask, fallback)

    dec_embed = model.token_embed(dec_in)
    dec_out = model.decoder(
        dec_embed,
        dec_mask,
        span_mem,
        mem_mask,
        train=True,
        key=chunk_key,
    )
    logits = model.token_embed.unembed(dec_out).astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_idx = dec_targets.astype(jnp.int32)
    token_logp = jnp.take_along_axis(log_probs, target_idx[..., None], axis=-1).squeeze(-1)
    loss = -token_logp * target_mask.astype(jnp.float32)
    loss_sum = jnp.sum(loss)
    count_sum = jnp.sum(target_mask.astype(jnp.float32))

    def _with_count() -> Array:
        return loss_sum / count_sum

    def _no_count() -> Array:
        return jnp.asarray(0.0, dtype=jnp.float32)

    return jax.lax.cond(count_sum > 0.0, _with_count, _no_count)


def _select_span_targets_per_view(
    sample_reps: Array,
    mask_positions: Array,
    *,
    base_norm: RMSNorm,
    key: Array,
) -> Array:
    """Select one masked-span target vector per sample and per view for SIGReg.

    :param sample_reps: Sample token reps of shape (B, T, D) pre-base-norm.
    :param mask_positions: Mask positions of shape (B, V, T) for global/local views.
    :param base_norm: Base model RMSNorm module.
    :param key: PRNG key for sampling.
    :returns: Array of shape (B, V, D) with one selected target per sample per view.
    """
    if sample_reps.ndim != 3:
        raise ValueError("sample_reps must have shape (B, T, D)")
    if mask_positions.ndim != 3:
        raise ValueError("mask_positions must have shape (B, V, T)")
    if sample_reps.shape[0] != mask_positions.shape[0]:
        raise ValueError("batch sizes must match")
    if sample_reps.shape[1] != mask_positions.shape[2]:
        raise ValueError("sequence lengths must match")

    num_segments = sample_reps.shape[1] + 1

    def _per_view(sample_rep: Array, mask: Array, view_key: Array) -> Array:
        span_ids = _span_ids(mask)
        sum_sample = jax.ops.segment_sum(sample_rep, span_ids, num_segments)
        counts = jax.ops.segment_sum(mask.astype(jnp.float32), span_ids, num_segments)
        denom = jnp.maximum(counts[:, None], 1.0)
        mean_sample = sum_sample / denom
        mean_sample = base_norm(mean_sample)
        valid = counts > 0
        valid = valid.at[0].set(False)
        valid_count = jnp.sum(valid)

        def _pick() -> Array:
            probs = valid.astype(jnp.float32)
            probs = probs / jnp.sum(probs)
            idx = jax.random.choice(view_key, valid.shape[0], p=probs)
            return mean_sample[idx]

        def _fallback() -> Array:
            return jnp.zeros((mean_sample.shape[-1],), dtype=mean_sample.dtype)

        return jax.lax.cond(valid_count > 0, _pick, _fallback)

    def _per_sample(sample_rep: Array, view_masks: Array, sample_key: Array) -> Array:
        view_keys = jax.random.split(sample_key, view_masks.shape[0])
        return jax.vmap(_per_view, in_axes=(None, 0, 0))(sample_rep, view_masks, view_keys)

    keys = jax.random.split(key, sample_reps.shape[0])
    return jax.vmap(_per_sample)(sample_reps, mask_positions, keys)


def _save_checkpoint(
    run_dir: str,
    step: int,
    model: TextTransformer,
    opt_state: eqx.Module,
    metadata: dict[str, object],
) -> None:
    """Save model, optimizer, and metadata to disk.

    :param run_dir: Run directory path.
    :param step: Global step index.
    :param model: Model to serialize.
    :param opt_state: Optimizer state to serialize.
    :param metadata: Metadata to persist.
    """
    ckpt_dir = os.path.join(run_dir, f"checkpoint_step_{step:08d}")
    os.makedirs(ckpt_dir, exist_ok=False)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "model.eqx"), model)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "optimizer.eqx"), opt_state)
    with open(os.path.join(ckpt_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    """Run the text training loop."""
    args = _parse_args()
    causal_attention = _parse_bool(args.causal_attention)

    if args.max_train_steps < 0:
        raise ValueError("max-train-steps must be >= 0")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad-accum-steps must be > 0")
    if args.log_every <= 0:
        raise ValueError("log-every must be > 0")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be > 0")
    if args.profile_warmup_steps < 0:
        raise ValueError("profile-warmup-steps must be >= 0")
    if args.num_global_views < 0:
        raise ValueError("num-global-views must be >= 0")
    if args.num_local_views < 0:
        raise ValueError("num-local-views must be >= 0")
    default_global_min = 0.25
    default_global_max = 0.30
    default_local_min = 0.40
    default_local_max = 0.45

    if args.global_mask_min is None:
        args.global_mask_min = default_global_min
    if args.global_mask_max is None:
        args.global_mask_max = default_global_max
    if args.local_mask_min is None:
        args.local_mask_min = default_local_min
    if args.local_mask_max is None:
        args.local_mask_max = default_local_max

    if args.global_mask_min < 0.0 or args.global_mask_min > 1.0:
        raise ValueError("global-mask-min must be in [0, 1]")
    if args.global_mask_max < 0.0 or args.global_mask_max > 1.0:
        raise ValueError("global-mask-max must be in [0, 1]")
    if args.local_mask_min < 0.0 or args.local_mask_min > 1.0:
        raise ValueError("local-mask-min must be in [0, 1]")
    if args.local_mask_max < 0.0 or args.local_mask_max > 1.0:
        raise ValueError("local-mask-max must be in [0, 1]")
    if args.global_mask_min > args.global_mask_max:
        raise ValueError("global-mask-min must be <= global-mask-max")
    if args.local_mask_min > args.local_mask_max:
        raise ValueError("local-mask-min must be <= local-mask-max")
    if args.sigreg_weight < 0.0 or args.sigreg_weight > 1.0:
        raise ValueError("sigreg-weight must be in [0, 1]")
    if args.pred_loss not in ("mse", "cosine"):
        raise ValueError("pred-loss must be 'mse' or 'cosine'")
    if args.seq_loss_weight < 0.0:
        raise ValueError("seq-loss-weight must be >= 0")
    if args.span_loss_weight < 0.0:
        raise ValueError("span-loss-weight must be >= 0")
    if args.decoder_loss_weight < 0.0:
        raise ValueError("decoder-loss-weight must be >= 0")
    if args.seq_loss_weight + args.span_loss_weight + args.decoder_loss_weight <= 0.0:
        raise ValueError("at least one loss weight must be > 0")
    if args.num_global_views + args.num_local_views <= 0:
        raise ValueError("at least one view must be requested")

    dtype = _dtype_from_name(args.dtype)
    if dtype in (jnp.bfloat16, jnp.float16):
        param_dtype = jnp.float32
    else:
        param_dtype = dtype
    betas = _parse_betas(args.adamw_betas)

    device_list: list[jax.Device] = _devices_for_platform("gpu")
    if args.num_devices == 0:
        device_list = _devices_for_platform("cpu")
    if args.num_devices > 0:
        if len(device_list) < args.num_devices:
            raise ValueError("not enough devices available")
        device_list = device_list[: args.num_devices]
    if len(device_list) == 0:
        raise ValueError("no devices available for training")

    num_devices = len(device_list)
    _mesh, data_sharding, _replicated_sharding = _build_sharding(device_list)

    dataset = MMapTokenDataset(args.data_folder)
    metadata = dataset.metadata()
    tokenizer_name = metadata.get("tokenizer")
    eos_token = metadata.get("eos_token")
    pad_token = metadata.get("pad_token")
    mask_token = metadata.get("mask_token")
    bos_token = metadata.get("bos_token")
    bos_id_raw = metadata.get("bos_id")
    if isinstance(tokenizer_name, str) is False:
        raise ValueError("metadata missing tokenizer")
    if isinstance(eos_token, str) is False:
        raise ValueError("metadata missing eos_token")
    if isinstance(pad_token, str) is False:
        raise ValueError("metadata missing pad_token")
    if isinstance(mask_token, str) is False:
        raise ValueError("metadata missing mask_token")
    max_seq_len = dataset.max_seq_len
    vocab_size, eos_id, pad_id, mask_id = dataset.tokenizer_info()
    total_samples = len(dataset)
    if isinstance(bos_token, str) is True and isinstance(bos_id_raw, int) is True:
        bos_id = int(bos_id_raw)
    else:
        bos_token = eos_token
        bos_id = eos_id

    if args.preview_views is True:
        tokenizer, _eos_id, _pad_id, _mask_id = _build_tokenizer(
            tokenizer_name,
            eos_token=eos_token,
            pad_token=pad_token,
            mask_token=mask_token,
        )
        global_batch = args.per_device_batch_size * num_devices
        preview_iter = iter_text_batches(
            dataset,
            batch_size=global_batch,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
        )
        batch_tokens = next(iter(preview_iter))
        batch_tokens = np.where(batch_tokens == eos_id, pad_id, batch_tokens)
        key = jax.random.PRNGKey(args.seed)
        key, global_key, local_key = jax.random.split(key, 3)
        global_views, _global_masks, _global_attn = _mask_views(
            jnp.asarray(batch_tokens),
            global_key,
            num_views=args.num_global_views,
            min_ratio=args.global_mask_min,
            max_ratio=args.global_mask_max,
            pad_id=pad_id,
            eos_id=eos_id,
            mask_id=mask_id,
            use_mask_token=args.mask_token_input,
        )
        local_views, _local_masks, _local_attn = _mask_views(
            jnp.asarray(batch_tokens),
            local_key,
            num_views=args.num_local_views,
            min_ratio=args.local_mask_min,
            max_ratio=args.local_mask_max,
            pad_id=pad_id,
            eos_id=eos_id,
            mask_id=mask_id,
            use_mask_token=args.mask_token_input,
        )

        for idx in range(batch_tokens.shape[0]):
            original = _decode_tokens(tokenizer, batch_tokens[idx], pad_id=pad_id)
            print(f"\nSample {idx + 1}")
            print("Original:")
            print(original)
            if args.num_global_views > 0:
                print("Global views:")
                for view_idx in range(args.num_global_views):
                    view_tokens = np.asarray(global_views[idx, view_idx])
                    view_mask = np.asarray(_global_masks[idx, view_idx])
                    view_tokens = np.where(view_mask, mask_id, view_tokens)
                    decoded = _decode_tokens(tokenizer, view_tokens, pad_id=pad_id)
                    print(f"  G{view_idx + 1}: {decoded}")
            if args.num_local_views > 0:
                print("Local views:")
                for view_idx in range(args.num_local_views):
                    view_tokens = np.asarray(local_views[idx, view_idx])
                    view_mask = np.asarray(_local_masks[idx, view_idx])
                    view_tokens = np.where(view_mask, mask_id, view_tokens)
                    decoded = _decode_tokens(tokenizer, view_tokens, pad_id=pad_id)
                    print(f"  L{view_idx + 1}: {decoded}")
        dataset.close()
        return

    run_dir = _build_run_dir(args.runs_folder)

    predictor_layers = args.predictor_n_layers
    if predictor_layers is None:
        predictor_layers = args.n_layers
    if predictor_layers <= 0:
        raise ValueError("predictor-n-layers must be > 0")
    decoder_layers = args.decoder_n_layers
    if decoder_layers is None:
        decoder_layers = args.n_layers
    if decoder_layers <= 0:
        raise ValueError("decoder-n-layers must be > 0")

    model_config = TextTransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        predictor_n_layers=int(predictor_layers),
        decoder_n_layers=int(decoder_layers),
        mlp_ratio=args.mlp_ratio,
        attn_dropout=args.attn_dropout,
        resid_dropout=args.resid_dropout,
        drop_path_rate=args.drop_path_rate,
        pope_base=args.pope_base,
        init_std=args.init_std,
        attn_type=args.attn_type,
        embed_norm=False,
        embed_norm_scale=args.init_std,
        causal_attention=causal_attention,
    )
    exclusion_patterns = list(TextTransformer.MUON_PARAM_EXCLUSION_PATTERNS)

    run_config: dict[str, object] = {
        "model": model_config.model_dump(),
        "data": {
            "dataset_folder": args.data_folder,
            "tokenizer": tokenizer_name,
            "eos_token": eos_token,
            "bos_token": bos_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "max_seq_len": max_seq_len,
            "shuffle_buffer": args.shuffle_buffer,
            "vocab_size": vocab_size,
            "eos_id": eos_id,
            "bos_id": bos_id,
            "pad_id": pad_id,
            "mask_id": mask_id,
            "num_samples": total_samples,
        },
        "views": {
            "num_global_views": args.num_global_views,
            "num_local_views": args.num_local_views,
            "global_mask_min": args.global_mask_min,
            "global_mask_max": args.global_mask_max,
            "local_mask_min": args.local_mask_min,
            "local_mask_max": args.local_mask_max,
            "mask_token_input": args.mask_token_input,
        },
        "loss": {
            "sigreg_weight": args.sigreg_weight,
            "sigreg_slices": args.sigreg_slices,
            "sigreg_seed": args.sigreg_seed,
            "pred_loss": args.pred_loss,
            "seq_loss_weight": args.seq_loss_weight,
            "span_loss_weight": args.span_loss_weight,
            "decoder_loss_weight": args.decoder_loss_weight,
            "encoder_mlm_weight": args.encoder_mlm_weight,
            "span_tokenwise": args.span_tokenwise,
        },
        "optimizer": {
            "muon_lr": args.muon_lr,
            "muon_momentum": args.muon_momentum,
            "muon_weight_decay": args.muon_weight_decay,
            "muon_nesterov": args.muon_nesterov,
            "muon_n_steps": args.muon_n_steps,
            "muon_ns_eps": args.muon_ns_eps,
            "adamw_lr": args.adamw_lr,
            "adamw_betas": betas,
            "adamw_eps": args.adamw_eps,
            "adamw_weight_decay": args.adamw_weight_decay,
            "muon_param_exclusion_patterns": exclusion_patterns,
        },
        "training": {
            "max_train_steps": args.max_train_steps,
            "grad_accum_steps": args.grad_accum_steps,
            "per_device_batch_size": args.per_device_batch_size,
            "num_devices": num_devices,
            "seed": args.seed,
            "dtype": args.dtype,
            "log_every": args.log_every,
            "checkpoint_every": args.checkpoint_every,
            "profile": args.profile,
            "profile_warmup_steps": args.profile_warmup_steps,
            "prefetch": args.prefetch,
        },
    }

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    writer = SummaryWriter(run_dir)
    writer.add_text("config", json.dumps(run_config, indent=2))

    base_key = jax.random.PRNGKey(args.seed)
    model_key, base_key = jax.random.split(base_key)
    model = TextTransformer(
        model_config,
        dtype=dtype,
        param_dtype=param_dtype,
        key=model_key,
    )
    params, static = eqx.partition(model, eqx.is_array)
    model_static = static
    flat_names = _flatten_param_names(params)
    muon_mask, qkv_mask = build_muon_masks(params, flat_names, exclusion_patterns)

    flat_params, _ = jax.tree_util.tree_flatten(params)
    flat_muon, _ = jax.tree_util.tree_flatten(muon_mask)
    if len(flat_params) != len(flat_muon):
        raise ValueError("muon_mask must align with params")

    muon_names = [
        name
        for name, param, flag in zip(flat_names, flat_params, flat_muon, strict=True)
        if isinstance(param, jax.Array) and flag
    ]
    adamw_names = [
        name
        for name, param, flag in zip(flat_names, flat_params, flat_muon, strict=True)
        if isinstance(param, jax.Array) and flag is False
    ]
    total_params = sum(int(param.size) for param in flat_params if isinstance(param, jax.Array))

    print(f"Total parameters: {total_params}")
    print("Muon parameters:")
    print("\n".join(muon_names))
    print("AdamW parameters:")
    print("\n".join(adamw_names))

    optimizer = MuonWithAdamWFallback(
        muon_learning_rate=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_weight_decay=args.muon_weight_decay,
        muon_nesterov=args.muon_nesterov,
        muon_n_steps=args.muon_n_steps,
        muon_ns_eps=args.muon_ns_eps,
        adamw_learning_rate=args.adamw_lr,
        adamw_betas=betas,
        adamw_eps=args.adamw_eps,
        adamw_weight_decay=args.adamw_weight_decay,
        muon_mask=muon_mask,
        qkv_mask=qkv_mask,
    )

    opt_state: MuonWithAdamWFallbackState = optimizer.init(params)

    def grad_step(
        model_in: TextTransformer,
        batch_tokens: Array,
        key: Array,
        global_step: Array,
        micro_step0: Array,
        micro_steps: Array,
    ) -> tuple[
        eqx.Module,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
    ]:
        """Compute accumulated gradients and metrics across micro-steps.

        :param model_in: Replicated model.
        :param batch_tokens: Token batches of shape (M, B, T).
        :param key: PRNG key for view masking.
        :param global_step: Global step index for loss scheduling.
        :param micro_step0: Starting micro-step index for RNG folding.
        :param micro_steps: Number of valid micro-steps in the batch.
        :returns: Tuple of (grads, total loss, seq rec loss, seq sigreg loss, span rec loss,
            span sigreg loss, seq lejepa loss, span lejepa loss, decoder loss,
            encoder mlm loss, encoder mlm acc1, encoder mlm acc5).
        """

        def _loss_fn(
            model_inner: TextTransformer,
            tokens: Array,
            tokens_key: Array,
        ) -> tuple[
            Array,
            tuple[
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
            ],
        ]:
            """Compute the total loss and its components.

            :param model_inner: Model replica used for the loss computation.
            :param tokens: Token batch of shape (B, T).
            :param tokens_key: PRNG key for masking.
            :returns: Tuple of (total loss, (seq rec loss, seq sigreg loss, span rec loss,
                span sigreg loss, seq lejepa loss, span lejepa loss, decoder loss,
                encoder mlm loss, encoder mlm acc1, encoder mlm acc5, total loss)).
            """
            view_key, predictor_key, sigreg_key, decoder_key = jax.random.split(tokens_key, 4)
            (
                pooled_post,
                pooled_pre,
                token_post,
                token_pre,
                mask_positions,
                view_attn,
            ) = _encode_views(
                model_inner,
                tokens,
                train=True,
                key=view_key,
                num_global_views=args.num_global_views,
                num_local_views=args.num_local_views,
                global_mask_min=args.global_mask_min,
                global_mask_max=args.global_mask_max,
                local_mask_min=args.local_mask_min,
                local_mask_max=args.local_mask_max,
                pad_id=pad_id,
                eos_id=eos_id,
                mask_id=mask_id,
                use_mask_token=args.mask_token_input,
            )

            total_views = args.num_global_views + args.num_local_views
            seq_rec_loss = jnp.asarray(0.0, dtype=jnp.float32)
            seq_sigreg_loss = jnp.asarray(0.0, dtype=jnp.float32)
            seq_lejepa_loss = jnp.asarray(0.0, dtype=jnp.float32)
            if args.seq_loss_weight > 0.0:
                pooled_post_views = pooled_post[:, :total_views, :]
                pooled_pre_global = pooled_pre[:, : args.num_global_views, :]
                center_pre = jnp.mean(pooled_pre_global, axis=1)
                center = model_inner.final_norm(center_pre)
                pooled_post_views = pooled_post_views.astype(jnp.float32)
                center = center.astype(jnp.float32)

                if args.pred_loss == "mse":
                    diffs = pooled_post_views - center[:, None, :]
                    seq_rec_loss = jnp.mean(jnp.square(diffs))
                else:
                    eps = jnp.asarray(1e-8, dtype=pooled_post_views.dtype)
                    emb_norm = pooled_post_views / (jnp.linalg.norm(pooled_post_views, axis=-1, keepdims=True) + eps)
                    ctr_norm = center / (jnp.linalg.norm(center, axis=-1, keepdims=True) + eps)
                    cos_sim = jnp.sum(emb_norm * ctr_norm[:, None, :], axis=-1)
                    seq_rec_loss = jnp.mean(1.0 - cos_sim)

                if args.sigreg_weight > 0.0:
                    seq_sigreg_loss = sigreg_loss_views(
                        pooled_post_views,
                        global_step=global_step,
                        num_slices=args.sigreg_slices,
                        seed=args.sigreg_seed,
                        axis_name="data",
                    )
                seq_lejepa_loss = (1.0 - args.sigreg_weight) * seq_rec_loss + args.sigreg_weight * seq_sigreg_loss

            bsz, num_views, seq_len, dim = token_post.shape
            batch_idx = jnp.arange(bsz, dtype=jnp.int32)
            view_tokens_list: list[Array] = []
            view_masks_list: list[Array] = []
            view_attn_list: list[Array] = []

            encoder_mlm_loss = jnp.asarray(0.0, dtype=jnp.float32)
            encoder_mlm_acc1 = jnp.asarray(0.0, dtype=jnp.float32)
            encoder_mlm_acc5 = jnp.asarray(0.0, dtype=jnp.float32)
            if args.encoder_mlm_weight > 0.0:
                total_views = args.num_global_views + args.num_local_views
                if total_views > 0:
                    mlm_reps = token_post[:, :total_views, :, :]
                    mlm_masks = mask_positions[:, :total_views, :]
                    tokens_no_eos = jnp.where(tokens == eos_id, pad_id, tokens)
                    reps = mlm_reps.reshape((bsz * total_views, seq_len, dim))
                    mask = mlm_masks.reshape((bsz * total_views, seq_len))
                    targets = jnp.repeat(tokens_no_eos, repeats=total_views, axis=0)

                    logits = model_inner.token_embed.unembed(reps)
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    target_logp = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
                    mask_f = mask.astype(jnp.float32)
                    count = jnp.sum(mask_f)
                    loss_sum = -jnp.sum(target_logp * mask_f)
                    encoder_mlm_loss = jnp.where(count > 0.0, loss_sum / count, 0.0)

                    preds = jnp.argmax(logits, axis=-1)
                    correct1 = jnp.logical_and(preds == targets, mask)
                    acc1_sum = jnp.sum(correct1.astype(jnp.float32))
                    encoder_mlm_acc1 = jnp.where(count > 0.0, acc1_sum / count, 0.0)

                    top5 = jax.lax.top_k(logits, k=5)[1]
                    correct5 = jnp.any(top5 == targets[..., None], axis=-1)
                    correct5 = jnp.logical_and(correct5, mask)
                    acc5_sum = jnp.sum(correct5.astype(jnp.float32))
                    encoder_mlm_acc5 = jnp.where(count > 0.0, acc5_sum / count, 0.0)

            if args.num_global_views > 0:
                global_key = jax.random.fold_in(predictor_key, 0)
                global_idx = jax.random.randint(
                    global_key,
                    shape=(bsz,),
                    minval=0,
                    maxval=args.num_global_views,
                )
                view_tokens_list.append(token_post[batch_idx, global_idx, :, :])
                view_masks_list.append(mask_positions[batch_idx, global_idx, :])
                view_attn_list.append(view_attn[batch_idx, global_idx, :])
            if args.num_local_views > 0:
                local_key = jax.random.fold_in(predictor_key, 1)
                local_idx = jax.random.randint(
                    local_key,
                    shape=(bsz,),
                    minval=0,
                    maxval=args.num_local_views,
                )
                local_idx = local_idx + args.num_global_views
                view_tokens_list.append(token_post[batch_idx, local_idx, :, :])
                view_masks_list.append(mask_positions[batch_idx, local_idx, :])
                view_attn_list.append(view_attn[batch_idx, local_idx, :])

            if len(view_tokens_list) == 0:
                raise ValueError("predictor requires at least one view")

            view_tokens_sel = jnp.stack(view_tokens_list, axis=1)
            view_masks_sel = jnp.stack(view_masks_list, axis=1)
            view_attn_sel = jnp.stack(view_attn_list, axis=1)

            span_sigreg_loss = jnp.asarray(0.0, dtype=jnp.float32)
            span_rec_loss = jnp.asarray(0.0, dtype=jnp.float32)
            span_lejepa_loss = jnp.asarray(0.0, dtype=jnp.float32)
            decoder_loss = jnp.asarray(0.0, dtype=jnp.float32)

            need_predictor = args.span_loss_weight > 0.0 or args.decoder_loss_weight > 0.0
            if need_predictor:
                if args.mask_token_input is True:
                    predictor_attn = view_attn_sel
                    predictor_mask = jnp.zeros_like(view_masks_sel)
                else:
                    predictor_attn = jnp.logical_or(view_attn_sel, view_masks_sel)
                    predictor_mask = view_masks_sel
                num_sel = view_tokens_sel.shape[1]
                flat_tokens = view_tokens_sel.reshape((bsz * num_sel, seq_len, dim))
                flat_attn = predictor_attn.reshape((bsz * num_sel, seq_len))
                flat_mask = predictor_mask.reshape((bsz * num_sel, seq_len))

                pred_reps = model_inner.predictor(
                    flat_tokens,
                    flat_attn,
                    flat_mask,
                    train=True,
                    key=predictor_key,
                )
                pred_reps = pred_reps.reshape((bsz, num_sel, seq_len, dim))

                view_pred_reps = pred_reps
                view_masks = view_masks_sel

                if args.span_loss_weight > 0.0:
                    sample_index = total_views
                    sample_reps = token_pre[:, sample_index, :, :]
                    if args.sigreg_weight > 0.0:
                        span_targets = _select_span_targets_per_view(
                            sample_reps,
                            mask_positions[:, :total_views, :],
                            base_norm=model_inner.final_norm,
                            key=sigreg_key,
                        )
                        span_sigreg_loss = sigreg_loss_views(
                            span_targets,
                            global_step=global_step,
                            num_slices=args.sigreg_slices,
                            seed=args.sigreg_seed,
                            axis_name="data",
                        )

                    if args.span_tokenwise is False:
                        span_rec_loss = _predictor_span_loss(
                            sample_reps,
                            view_pred_reps,
                            view_masks,
                            base_norm=model_inner.final_norm,
                            predictor_norm=model_inner.predictor.final_norm,
                        )
                    else:
                        span_rec_loss = _predictor_span_loss_tokenwise(
                            sample_reps,
                            view_pred_reps,
                            view_masks,
                            base_norm=model_inner.final_norm,
                            predictor_norm=model_inner.predictor.final_norm,
                        )
                    span_lejepa_loss = (
                        (1.0 - args.sigreg_weight) * span_rec_loss + args.sigreg_weight * span_sigreg_loss
                    )

                if args.decoder_loss_weight > 0.0:
                    tokens_no_eos = jnp.where(tokens == eos_id, pad_id, tokens)
                    decoder_loss = _decoder_loss(
                        model_inner,
                        tokens_no_eos,
                        view_pred_reps,
                        view_masks,
                        bos_id=bos_id,
                        eos_id=eos_id,
                        pad_id=pad_id,
                        key=decoder_key,
                    )

            seq_weight = jnp.asarray(args.seq_loss_weight, dtype=jnp.float32)
            span_weight = jnp.asarray(args.span_loss_weight, dtype=jnp.float32)
            decoder_weight = jnp.asarray(args.decoder_loss_weight, dtype=jnp.float32)
            mlm_weight = jnp.asarray(args.encoder_mlm_weight, dtype=jnp.float32)
            weight_sum = seq_weight + span_weight + decoder_weight + mlm_weight
            weight_sum = jnp.maximum(weight_sum, 1e-6)
            seq_frac = seq_weight / weight_sum
            span_frac = span_weight / weight_sum
            decoder_frac = decoder_weight / weight_sum
            mlm_frac = mlm_weight / weight_sum
            total_loss = (
                seq_frac * seq_lejepa_loss
                + span_frac * span_lejepa_loss
                + decoder_frac * decoder_loss
                + mlm_frac * encoder_mlm_loss
            )
            return (
                total_loss,
                (
                    seq_rec_loss,
                    seq_sigreg_loss,
                    span_rec_loss,
                    span_sigreg_loss,
                    seq_lejepa_loss,
                    span_lejepa_loss,
                    decoder_loss,
                    encoder_mlm_loss,
                    encoder_mlm_acc1,
                    encoder_mlm_acc5,
                    total_loss,
                ),
            )

        value_and_grad = eqx.filter_value_and_grad(_loss_fn, has_aux=True)
        params_only = eqx.filter(model_in, eqx.is_array)
        grad_init = jax.tree_util.tree_map(
            lambda value: jnp.zeros_like(value) if value is not None else None,
            params_only,
        )
        loss_init = jnp.asarray(0.0, dtype=jnp.float32)
        seq_rec_init = jnp.asarray(0.0, dtype=jnp.float32)
        seq_sigreg_init = jnp.asarray(0.0, dtype=jnp.float32)
        span_rec_init = jnp.asarray(0.0, dtype=jnp.float32)
        span_sigreg_init = jnp.asarray(0.0, dtype=jnp.float32)
        seq_lejepa_init = jnp.asarray(0.0, dtype=jnp.float32)
        span_lejepa_init = jnp.asarray(0.0, dtype=jnp.float32)
        decoder_loss_init = jnp.asarray(0.0, dtype=jnp.float32)
        encoder_mlm_init = jnp.asarray(0.0, dtype=jnp.float32)
        encoder_mlm_acc1_init = jnp.asarray(0.0, dtype=jnp.float32)
        encoder_mlm_acc5_init = jnp.asarray(0.0, dtype=jnp.float32)

        step_indices = jnp.arange(batch_tokens.shape[0], dtype=jnp.int32)

        def _accum_body(
            carry: tuple[eqx.Module, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
            inputs: tuple[Array, Array],
        ) -> tuple[
            tuple[eqx.Module, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
            None,
        ]:
            """Accumulate gradients and metrics for one micro-step.

            :param carry: Tuple of (grads, total loss, seq rec loss, seq sigreg loss, span rec loss,
                span sigreg loss, seq lejepa loss, span lejepa loss, decoder loss,
                encoder mlm loss, encoder mlm acc1, encoder mlm acc5).
            :param inputs: Tuple of (micro step index, token batch).
            :returns: Updated carry and unused output.
            """
            (
                grads_acc,
                loss_acc,
                seq_rec_acc,
                seq_sigreg_acc,
                span_rec_acc,
                span_sigreg_acc,
                seq_lejepa_acc,
                span_lejepa_acc,
                decoder_loss_acc,
                encoder_mlm_acc,
                encoder_mlm_acc1_acc,
                encoder_mlm_acc5_acc,
            ) = carry
            step_idx, tokens = inputs
            step_idx_global = micro_step0 + step_idx
            tokens_key = jax.random.fold_in(key, step_idx_global)

            def _do_compute(
                _: None,
            ) -> tuple[eqx.Module, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
                (
                    loss,
                    (
                        seq_rec_loss,
                        seq_sigreg_loss,
                        span_rec_loss,
                        span_sigreg_loss,
                        seq_lejepa_loss,
                        span_lejepa_loss,
                        decoder_loss,
                        encoder_mlm_loss,
                        encoder_mlm_acc1,
                        encoder_mlm_acc5,
                        _total_loss,
                    ),
                ), grads = value_and_grad(
                    model_in,
                    tokens,
                    tokens_key,
                )
                grads = _cast_tree_dtype(grads, jnp.float32)
                grads_accum = _add_trees(grads_acc, grads)
                loss_accum = loss_acc + loss
                seq_rec_accum = seq_rec_acc + seq_rec_loss
                seq_sigreg_accum = seq_sigreg_acc + seq_sigreg_loss
                span_rec_accum = span_rec_acc + span_rec_loss
                span_sigreg_accum = span_sigreg_acc + span_sigreg_loss
                seq_lejepa_accum = seq_lejepa_acc + seq_lejepa_loss
                span_lejepa_accum = span_lejepa_acc + span_lejepa_loss
                decoder_loss_accum = decoder_loss_acc + decoder_loss
                encoder_mlm_accum = encoder_mlm_acc + encoder_mlm_loss
                encoder_mlm_acc1_accum = encoder_mlm_acc1_acc + encoder_mlm_acc1
                encoder_mlm_acc5_accum = encoder_mlm_acc5_acc + encoder_mlm_acc5
                return (
                    grads_accum,
                    loss_accum,
                    seq_rec_accum,
                    seq_sigreg_accum,
                    span_rec_accum,
                    span_sigreg_accum,
                    seq_lejepa_accum,
                    span_lejepa_accum,
                    decoder_loss_accum,
                    encoder_mlm_accum,
                    encoder_mlm_acc1_accum,
                    encoder_mlm_acc5_accum,
                )

            def _skip_compute(
                _: None,
            ) -> tuple[eqx.Module, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
                return (
                    grads_acc,
                    loss_acc,
                    seq_rec_acc,
                    seq_sigreg_acc,
                    span_rec_acc,
                    span_sigreg_acc,
                    seq_lejepa_acc,
                    span_lejepa_acc,
                    decoder_loss_acc,
                    encoder_mlm_acc,
                    encoder_mlm_acc1_acc,
                    encoder_mlm_acc5_acc,
                )

            active = step_idx < micro_steps
            new_carry = jax.lax.cond(active, _do_compute, _skip_compute, operand=None)
            return new_carry, None

        (
            grads,
            loss,
            seq_rec_loss,
            seq_sigreg_loss,
            span_rec_loss,
            span_sigreg_loss,
            seq_lejepa_loss,
            span_lejepa_loss,
            decoder_loss,
            encoder_mlm_loss,
            encoder_mlm_acc1,
            encoder_mlm_acc5,
        ), _ = jax.lax.scan(
            _accum_body,
            (
                grad_init,
                loss_init,
                seq_rec_init,
                seq_sigreg_init,
                span_rec_init,
                span_sigreg_init,
                seq_lejepa_init,
                span_lejepa_init,
                decoder_loss_init,
                encoder_mlm_init,
                encoder_mlm_acc1_init,
                encoder_mlm_acc5_init,
            ),
            (step_indices, batch_tokens),
        )

        scale = jnp.asarray(micro_steps, dtype=jnp.float32)
        scale = jnp.maximum(scale, 1.0)
        grads = jax.tree_util.tree_map(
            lambda value: value / scale if value is not None else None,
            grads,
        )
        loss = loss / scale
        seq_rec_loss = seq_rec_loss / scale
        seq_sigreg_loss = seq_sigreg_loss / scale
        span_rec_loss = span_rec_loss / scale
        span_sigreg_loss = span_sigreg_loss / scale
        seq_lejepa_loss = seq_lejepa_loss / scale
        span_lejepa_loss = span_lejepa_loss / scale
        decoder_loss = decoder_loss / scale
        encoder_mlm_loss = encoder_mlm_loss / scale
        encoder_mlm_acc1 = encoder_mlm_acc1 / scale
        encoder_mlm_acc5 = encoder_mlm_acc5 / scale

        grads = jax.lax.pmean(grads, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")
        seq_rec_loss = jax.lax.pmean(seq_rec_loss, axis_name="data")
        seq_sigreg_loss = jax.lax.pmean(seq_sigreg_loss, axis_name="data")
        span_rec_loss = jax.lax.pmean(span_rec_loss, axis_name="data")
        span_sigreg_loss = jax.lax.pmean(span_sigreg_loss, axis_name="data")
        seq_lejepa_loss = jax.lax.pmean(seq_lejepa_loss, axis_name="data")
        span_lejepa_loss = jax.lax.pmean(span_lejepa_loss, axis_name="data")
        decoder_loss = jax.lax.pmean(decoder_loss, axis_name="data")
        encoder_mlm_loss = jax.lax.pmean(encoder_mlm_loss, axis_name="data")
        encoder_mlm_acc1 = jax.lax.pmean(encoder_mlm_acc1, axis_name="data")
        encoder_mlm_acc5 = jax.lax.pmean(encoder_mlm_acc5, axis_name="data")
        return (
            grads,
            loss,
            seq_rec_loss,
            seq_sigreg_loss,
            span_rec_loss,
            span_sigreg_loss,
            seq_lejepa_loss,
            span_lejepa_loss,
            decoder_loss,
            encoder_mlm_loss,
            encoder_mlm_acc1,
            encoder_mlm_acc5,
        )

    def apply_step(
        model_in: TextTransformer,
        state_in: MuonWithAdamWFallbackState,
        grads: eqx.Module,
    ) -> tuple[TextTransformer, MuonWithAdamWFallbackState]:
        """Apply gradients to the model.

        :param model_in: Replicated model.
        :param state_in: Replicated optimizer state.
        :param grads: Gradient PyTree for the model.
        :returns: Updated model and optimizer state.
        """
        params_inner = eqx.filter(model_in, eqx.is_array)
        updates, new_state = optimizer.update(grads, state_in, params_inner)
        new_model = eqx.apply_updates(model_in, updates)
        return new_model, new_state

    grad_step_pmap = eqx.filter_pmap(
        grad_step,
        axis_name="data",
        devices=device_list,
    )  # type: ignore[call-overload]
    apply_step_pmap = eqx.filter_pmap(
        apply_step,
        axis_name="data",
        devices=device_list,
        donate="all",
    )  # type: ignore[call-overload]

    params_repl = _replicate_tree(params, num_devices)
    params_repl = jax.device_put(params_repl, device=data_sharding)
    train_repl = eqx.combine(params_repl, model_static)
    opt_state_repl = _replicate_tree(opt_state, num_devices)
    opt_state_repl = jax.device_put(opt_state_repl, device=data_sharding)

    global_batch = args.per_device_batch_size * num_devices
    global_step = 0
    perf_steps = 0
    perf_data_time = 0.0
    perf_compute_time = 0.0
    perf_log_time = 0.0
    perf_warmup = args.profile_warmup_steps
    last_loss_val = 0.0
    last_seq_rec_val = 0.0
    last_seq_sigreg_val = 0.0
    last_span_rec_val = 0.0
    last_span_sigreg_val = 0.0
    last_seq_lejepa_val = 0.0
    last_span_lejepa_val = 0.0
    last_decoder_val = 0.0
    last_encoder_mlm_val = 0.0
    last_encoder_mlm_acc1_val = 0.0
    last_encoder_mlm_acc5_val = 0.0
    if total_samples <= 0:
        raise ValueError("no valid samples found in the dataset")
    epoch_steps = total_samples // global_batch
    if epoch_steps <= 0:
        raise ValueError("dataset too small for the configured batch size")
    if args.max_train_steps > 0:
        total_micro_steps = min(epoch_steps, args.max_train_steps * args.grad_accum_steps)
    else:
        total_micro_steps = epoch_steps
    total_steps = int(math.ceil(total_micro_steps / args.grad_accum_steps))

    try:
        train_iter_host = _iter_batches(
            dataset,
            batch_size=global_batch,
            max_seq_len=max_seq_len,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
            num_devices=num_devices,
            per_device_batch=args.per_device_batch_size,
        )
        train_iter = _prefetch_to_device(
            train_iter_host,
            size=args.prefetch,
            sharding=data_sharding,
        )
        train_iter = iter(train_iter)

        micro_step = 0
        with tqdm(total=total_steps, desc="Train") as pbar:
            for _ in range(total_steps):
                if micro_step >= total_micro_steps:
                    break
                micro_steps = min(args.grad_accum_steps, total_micro_steps - micro_step)
                if micro_steps <= 0:
                    break
                step_start = time.perf_counter()
                data_time = 0.0
                compute_time = 0.0
                step_key = jax.random.fold_in(base_key, global_step)
                step_id = jnp.full((num_devices,), global_step, dtype=jnp.int32)
                micro_step0 = jnp.full((num_devices,), micro_step, dtype=jnp.int32)
                micro_steps_arr = jnp.full((num_devices,), micro_steps, dtype=jnp.int32)
                micro_batches: list[Array] = []
                last_batch: Array | None = None

                for _micro_idx in range(micro_steps):
                    fetch_start = time.perf_counter()
                    batch_tokens = next(train_iter)
                    data_time += time.perf_counter() - fetch_start
                    micro_batches.append(batch_tokens)
                    last_batch = batch_tokens

                if len(micro_batches) < args.grad_accum_steps:
                    if last_batch is None:
                        break
                    pad_count = args.grad_accum_steps - len(micro_batches)
                    for _ in range(pad_count):
                        micro_batches.append(last_batch)

                batch_tokens = jnp.stack(micro_batches, axis=0)
                batch_tokens = jnp.transpose(batch_tokens, (1, 0, 2, 3))
                batch_tokens = jax.device_put(batch_tokens, device=data_sharding)
                device_keys = jax.random.split(step_key, num_devices)
                device_keys = jax.device_put(device_keys, device=data_sharding)

                compute_start = time.perf_counter()
                (
                    grads,
                    loss,
                    seq_rec,
                    seq_sigreg,
                    span_rec,
                    span_sigreg,
                    seq_lejepa,
                    span_lejepa,
                    decoder_loss,
                    encoder_mlm_loss,
                    encoder_mlm_acc1,
                    encoder_mlm_acc5,
                ) = grad_step_pmap(
                    train_repl,
                    batch_tokens,
                    device_keys,
                    step_id,
                    micro_step0,
                    micro_steps_arr,
                )
                compute_time += time.perf_counter() - compute_start
                micro_step += micro_steps

                compute_start = time.perf_counter()
                train_repl, opt_state_repl = apply_step_pmap(
                    train_repl,
                    opt_state_repl,
                    grads,
                )
                compute_time += time.perf_counter() - compute_start
                grads = None

                if args.profile:
                    jax.block_until_ready(loss)

                log_start = time.perf_counter()
                log_this_step = (global_step % args.log_every) == 0
                if log_this_step:
                    loss_val = float(np.mean(jax.device_get(loss)))
                    seq_rec_val = float(np.mean(jax.device_get(seq_rec)))
                    seq_sigreg_val = float(np.mean(jax.device_get(seq_sigreg)))
                    span_rec_val = float(np.mean(jax.device_get(span_rec)))
                    span_sigreg_val = float(np.mean(jax.device_get(span_sigreg)))
                    seq_lejepa_val = float(np.mean(jax.device_get(seq_lejepa)))
                    span_lejepa_val = float(np.mean(jax.device_get(span_lejepa)))
                    decoder_val = float(np.mean(jax.device_get(decoder_loss)))
                    encoder_mlm_val = float(np.mean(jax.device_get(encoder_mlm_loss)))
                    encoder_mlm_acc1_val = float(np.mean(jax.device_get(encoder_mlm_acc1)))
                    encoder_mlm_acc5_val = float(np.mean(jax.device_get(encoder_mlm_acc5)))
                    last_loss_val = loss_val
                    last_seq_rec_val = seq_rec_val
                    last_seq_sigreg_val = seq_sigreg_val
                    last_span_rec_val = span_rec_val
                    last_span_sigreg_val = span_sigreg_val
                    last_seq_lejepa_val = seq_lejepa_val
                    last_span_lejepa_val = span_lejepa_val
                    last_decoder_val = decoder_val
                    last_encoder_mlm_val = encoder_mlm_val
                    last_encoder_mlm_acc1_val = encoder_mlm_acc1_val
                    last_encoder_mlm_acc5_val = encoder_mlm_acc5_val

                    writer.add_scalar("train/total_loss", loss_val, global_step)
                    writer.add_scalar("train/seq_rec_loss", seq_rec_val, global_step)
                    writer.add_scalar("train/seq_sigreg_loss", seq_sigreg_val, global_step)
                    writer.add_scalar("train/span_rec_loss", span_rec_val, global_step)
                    writer.add_scalar("train/span_sigreg_loss", span_sigreg_val, global_step)
                    writer.add_scalar("train/seq_lejepa_loss", seq_lejepa_val, global_step)
                    writer.add_scalar("train/span_lejepa_loss", span_lejepa_val, global_step)
                    writer.add_scalar("train/decoder_loss", decoder_val, global_step)
                    writer.add_scalar("train/encoder_mlm_loss", encoder_mlm_val, global_step)
                    writer.add_scalar("train/encoder_mlm_acc1", encoder_mlm_acc1_val, global_step)
                    writer.add_scalar("train/encoder_mlm_acc5", encoder_mlm_acc5_val, global_step)

                if log_this_step:
                    pbar.set_postfix(
                        total=f"{last_loss_val:.4f}",
                        seq_rec=f"{last_seq_rec_val:.4f}",
                        span_rec=f"{last_span_rec_val:.4f}",
                        seq_sig=f"{last_seq_sigreg_val:.4f}",
                        span_sig=f"{last_span_sigreg_val:.4f}",
                        seq_ljp=f"{last_seq_lejepa_val:.4f}",
                        span_ljp=f"{last_span_lejepa_val:.4f}",
                        dec=f"{last_decoder_val:.4f}",
                        mlm=f"{last_encoder_mlm_val:.4f}",
                        mlm1=f"{last_encoder_mlm_acc1_val:.4f}",
                        mlm5=f"{last_encoder_mlm_acc5_val:.4f}",
                    )
                pbar.update(1)
                global_step += 1
                log_done = time.perf_counter()

                if args.profile:
                    if perf_steps + perf_warmup < global_step:
                        perf_steps += 1
                        perf_data_time += data_time
                        perf_compute_time += compute_time
                        perf_log_time += log_done - log_start

                if global_step % args.checkpoint_every == 0:
                    jax.block_until_ready(loss)
                    _block_until_ready(train_repl)
                    _block_until_ready(opt_state_repl)
                    opt_state_host = cast(MuonWithAdamWFallbackState, _unreplicate(opt_state_repl))
                    opt_state_host = cast(MuonWithAdamWFallbackState, _to_host(opt_state_host))
                    model_host = cast(TextTransformer, _unreplicate(train_repl))
                    model_host = cast(TextTransformer, _to_host(model_host))
                    metadata = {
                        "global_step": global_step,
                        "config": run_config,
                    }
                    _save_checkpoint(
                        run_dir,
                        global_step,
                        model_host,
                        opt_state_host,
                        metadata,
                    )
                    writer.flush()
                    if args.profile and perf_steps > 0:
                        total_time = perf_data_time + perf_compute_time + perf_log_time
                        if total_time > 0.0:
                            step_time = total_time / perf_steps
                            steps_per_sec = perf_steps / total_time
                            data_pct = 100.0 * perf_data_time / total_time
                            compute_pct = 100.0 * perf_compute_time / total_time
                            log_pct = 100.0 * perf_log_time / total_time
                            print(
                                "Perf summary (train): "
                                f"steps={perf_steps}, "
                                f"step_time={step_time:.4f}s, "
                                f"steps_per_sec={steps_per_sec:.2f}, "
                                f"data={data_pct:.1f}%, "
                                f"compute={compute_pct:.1f}%, "
                                f"log={log_pct:.1f}%"
                            )
                        perf_steps = 0
                        perf_data_time = 0.0
                        perf_compute_time = 0.0
                        perf_log_time = 0.0
    finally:
        dataset.close()
        writer.close()


if __name__ == "__main__":
    main()
