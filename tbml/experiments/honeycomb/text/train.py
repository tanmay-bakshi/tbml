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

from tbml.experiments.honeycomb.loss import sigreg_loss_views_masked
from tbml.experiments.honeycomb.text.dataset import (
    MMapTokenDataset,
    _build_tokenizer,
    iter_text_batches,
)
from tbml.experiments.honeycomb.text.model import TextTransformer, TextTransformerConfig
from tbml.optimizers import (
    MuonWithAdamWFallback,
    MuonWithAdamWFallbackState,
    build_muon_masks,
    build_weight_decay_mask,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train a text encoder with data2vec-style objectives.")
    parser.add_argument("--runs-folder", type=str, required=True)
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
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
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--resid-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--pope-base", type=float, default=10000.0)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--attn-type", type=str, default="pope", choices=["pope", "rope"])

    parser.add_argument("--num-views", type=int, default=6)
    parser.add_argument("--mask-ratio", type=float, default=0.42)
    parser.add_argument("--mask-ratio-adjust", type=float, default=0.1)
    parser.add_argument("--mask-block-size", type=int, default=3)

    parser.add_argument("--data2vec-loss-weight", type=float, default=1.0)
    parser.add_argument("--teacher-mode", type=str, default="ema", choices=["ema", "swa", "none"])
    parser.add_argument("--teacher-ema-start", type=float, default=0.9999)
    parser.add_argument("--teacher-ema-end", type=float, default=1.0)
    parser.add_argument("--teacher-ema-steps", type=int, default=100000)
    parser.add_argument("--teacher-top-k", type=int, default=12)
    parser.add_argument("--teacher-instance-norm", dest="teacher_instance_norm", action="store_true")
    parser.add_argument("--no-teacher-instance-norm", dest="teacher_instance_norm", action="store_false")
    parser.add_argument("--encoder-mlm-loss-weight", type=float, default=0.0)
    parser.add_argument("--encoder-mlm-keep-prob", type=float, default=0.0)
    parser.add_argument("--predictor-keep-unmasked", action="store_true")
    parser.add_argument("--sigreg-weight", type=float, default=0.0)
    parser.add_argument("--sigreg-slices", type=int, default=256)
    parser.add_argument("--sigreg-seed", type=int, default=0)
    parser.add_argument("--sigreg-student", dest="sigreg_student", action="store_true")
    parser.add_argument("--no-sigreg-student", dest="sigreg_student", action="store_false")
    parser.add_argument("--sigreg-mean-subtract", dest="sigreg_mean_subtract", action="store_true")
    parser.add_argument("--no-sigreg-mean-subtract", dest="sigreg_mean_subtract", action="store_false")

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

    parser.set_defaults(
        muon_nesterov=True,
        teacher_instance_norm=True,
        sigreg_student=True,
        sigreg_mean_subtract=True,
    )
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


def _clip_grad_norm(tree: T, max_norm: float) -> tuple[T, Array]:
    """Clip gradients by global norm.

    :param tree: Gradient PyTree containing arrays or None entries.
    :param max_norm: Maximum allowed L2 norm.
    :returns: Tuple of (clipped gradients, pre-clip norm).
    """
    if max_norm <= 0.0:
        return tree, jnp.asarray(0.0, dtype=jnp.float32)

    def _leaf_sqsum(value: object) -> Array:
        if value is None:
            return jnp.asarray(0.0, dtype=jnp.float32)
        if isinstance(value, jax.Array):
            return jnp.sum(jnp.square(value.astype(jnp.float32)))
        return jnp.asarray(0.0, dtype=jnp.float32)

    sums = jax.tree_util.tree_map(_leaf_sqsum, tree)
    total = jax.tree_util.tree_reduce(
        lambda acc, val: acc + val,
        sums,
        initializer=jnp.asarray(0.0, dtype=jnp.float32),
    )
    norm = jnp.sqrt(total)
    max_norm_f = jnp.asarray(max_norm, dtype=jnp.float32)
    denom = norm + jnp.asarray(1e-6, dtype=jnp.float32)
    scale = jnp.minimum(jnp.asarray(1.0, dtype=jnp.float32), max_norm_f / denom)
    clipped = cast(
        T,
        jax.tree_util.tree_map(
            lambda value: value * scale if value is not None else None,
            tree,
        ),
    )
    return clipped, norm


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


def _inverse_block_mask(
    tokens: Array,
    key: Array,
    *,
    mask_ratio: float,
    mask_ratio_adjust: float,
    block_size: int,
    pad_id: int,
    eos_id: int,
) -> tuple[Array, Array, Array]:
    """Apply inverse-block masking to tokens.

    Masked tokens are excluded from attention but the sequence length is unchanged.

    :param tokens: Token ids of shape (B, T).
    :param key: PRNG key.
    :param mask_ratio: Masking ratio in [0, 1].
    :param mask_ratio_adjust: Adjustment factor for the number of preserved blocks.
    :param block_size: Size of preserved blocks.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :returns: Tuple of (token ids, mask positions, attention mask).
    """
    if mask_ratio < 0.0 or mask_ratio > 1.0:
        raise ValueError("mask_ratio must be in [0, 1]")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if mask_ratio_adjust < 0.0 or mask_ratio_adjust > 1.0:
        raise ValueError("mask_ratio_adjust must be in [0, 1]")

    seq_len = tokens.shape[1]
    keys = jax.random.split(key, tokens.shape[0])
    positions = jnp.arange(seq_len, dtype=jnp.int32)
    left = block_size // 2
    right = block_size - left - 1

    def _one_sample(
        sample_tokens: Array,
        sample_key: Array,
    ) -> tuple[Array, Array, Array]:
        """Mask tokens for a single sequence using inverse-block masking.

        :param sample_tokens: Token ids of shape (T,).
        :param sample_key: PRNG key.
        :returns: Tuple of (masked tokens, mask positions, attention mask).
        """
        key_starts, key_drop, key_add = jax.random.split(sample_key, 3)
        eligible = jnp.logical_and(sample_tokens != pad_id, sample_tokens != eos_id)
        valid_len = jnp.sum(eligible).astype(jnp.int32)
        target_keep = jnp.floor((1.0 - mask_ratio) * valid_len).astype(jnp.int32)
        target_keep = jnp.minimum(target_keep, valid_len)
        target_keep = jnp.maximum(target_keep, 1)
        target_keep = jnp.where(valid_len > 0, target_keep, 0)

        starts_f = valid_len.astype(jnp.float32) * (
            (1.0 - mask_ratio) + mask_ratio_adjust
        )
        starts_f = starts_f / float(block_size)
        num_starts = jnp.floor(starts_f).astype(jnp.int32)
        num_starts = jnp.minimum(num_starts, valid_len)
        num_starts = jnp.maximum(num_starts, jnp.where(valid_len > 0, 1, 0))

        start_scores = jax.random.uniform(key_starts, shape=(seq_len,), minval=0.0, maxval=1.0)
        start_scores = jnp.where(eligible, start_scores, -jnp.inf)
        order = jnp.argsort(-start_scores)
        ranks = jnp.argsort(order)
        start_mask = ranks < num_starts

        pos = positions[:, None]
        starts = positions[None, :]
        in_block = jnp.logical_and(pos >= starts - left, pos <= starts + right)
        active = start_mask[None, :]
        keep_mask = jnp.any(jnp.logical_and(in_block, active), axis=1)
        keep_mask = jnp.logical_and(keep_mask, eligible)

        keep_count = jnp.sum(keep_mask).astype(jnp.int32)

        def _drop(_: None) -> Array:
            scores = jax.random.uniform(key_drop, shape=(seq_len,), minval=0.0, maxval=1.0)
            scores = jnp.where(keep_mask, scores, -jnp.inf)
            order = jnp.argsort(-scores)
            ranks = jnp.argsort(order)
            new_keep = ranks < target_keep
            return jnp.logical_and(new_keep, keep_mask)

        def _no_drop(_: None) -> Array:
            return keep_mask

        keep_mask = jax.lax.cond(keep_count > target_keep, _drop, _no_drop, operand=None)
        keep_count = jnp.sum(keep_mask).astype(jnp.int32)
        add_count = target_keep - keep_count

        def _add(_: None) -> Array:
            scores = jax.random.uniform(key_add, shape=(seq_len,), minval=0.0, maxval=1.0)
            available = jnp.logical_and(eligible, jnp.logical_not(keep_mask))
            scores = jnp.where(available, scores, -jnp.inf)
            order = jnp.argsort(-scores)
            ranks = jnp.argsort(order)
            add_mask = ranks < add_count
            add_mask = jnp.logical_and(add_mask, available)
            return jnp.logical_or(keep_mask, add_mask)

        def _no_add(_: None) -> Array:
            return keep_mask

        keep_mask = jax.lax.cond(add_count > 0, _add, _no_add, operand=None)
        mask = jnp.logical_and(eligible, jnp.logical_not(keep_mask))
        attention_mask = keep_mask

        def _ensure_any(attn: Array, elig: Array) -> Array:
            def _with_fallback(_: None) -> Array:
                fallback_idx = jnp.argmax(elig)
                return attn.at[fallback_idx].set(True)

            def _no_fallback(_: None) -> Array:
                return attn

            return jax.lax.cond(jnp.any(attn), _no_fallback, _with_fallback, operand=None)

        attention_mask = _ensure_any(attention_mask, eligible)
        return sample_tokens, mask, attention_mask

    masked, positions, attn_mask = jax.vmap(_one_sample)(tokens, keys)
    return masked, positions, attn_mask


def _mask_views(
    tokens: Array,
    key: Array,
    *,
    num_views: int,
    mask_ratio: float,
    mask_ratio_adjust: float,
    mask_block_size: int,
    pad_id: int,
    eos_id: int,
) -> tuple[Array, Array, Array]:
    """Generate multiple masked views of the token batch.

    :param tokens: Token ids of shape (B, T).
    :param key: PRNG key.
    :param num_views: Number of views to generate.
    :param mask_ratio: Masking ratio for inverse-block masking.
    :param mask_ratio_adjust: Adjustment factor for inverse-block masking.
    :param mask_block_size: Block size for inverse-block masking.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
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
        return _inverse_block_mask(
            tokens,
            view_key,
            mask_ratio=mask_ratio,
            mask_ratio_adjust=mask_ratio_adjust,
            block_size=mask_block_size,
            pad_id=pad_id,
            eos_id=eos_id,
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
    num_views: int,
    mask_ratio: float,
    mask_ratio_adjust: float,
    mask_block_size: int,
    pad_id: int,
    eos_id: int,
) -> tuple[Array, Array, Array, Array]:
    """Build masked views for multi-mask training.

    :param tokens: Token ids of shape (B, T).
    :param key: PRNG key.
    :param num_views: Number of masked views to generate per sample.
    :param mask_ratio: Masking ratio in [0, 1].
    :param mask_ratio_adjust: Adjustment factor for inverse-block masking.
    :param mask_block_size: Block size for inverse-block masking.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :returns: Tuple of (model key, views, masks, attention masks).
    """
    model_key, view_key = jax.random.split(key, 2)
    views, masks, attn = _mask_views(
        tokens,
        view_key,
        num_views=num_views,
        mask_ratio=mask_ratio,
        mask_ratio_adjust=mask_ratio_adjust,
        mask_block_size=mask_block_size,
        pad_id=pad_id,
        eos_id=eos_id,
    )
    return model_key, views, masks, attn


def _encode_views(
    model: TextTransformer,
    tokens: Array,
    *,
    train: bool,
    key: Array,
    num_views: int,
    mask_ratio: float,
    mask_ratio_adjust: float,
    mask_block_size: int,
    pad_id: int,
    eos_id: int,
) -> tuple[Array, Array, Array, Array]:
    """Encode masked views into token representations.

    :param model: Text transformer encoder.
    :param tokens: Token ids of shape (B, T).
    :param train: Whether to enable dropout.
    :param key: PRNG key.
    :param num_views: Number of masked views per sample.
    :param mask_ratio: Masking ratio in [0, 1].
    :param mask_ratio_adjust: Adjustment factor for inverse-block masking.
    :param mask_block_size: Block size for inverse-block masking.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :returns: Tuple of (views, token_post, mask_positions, view_attn).
    """
    if num_views <= 0:
        raise ValueError("at least one view must be requested")

    tokens = jnp.where(tokens == eos_id, jnp.asarray(pad_id, dtype=tokens.dtype), tokens)

    (
        model_key,
        views,
        mask_positions,
        view_attn,
    ) = _build_views(
        tokens,
        key,
        num_views=num_views,
        mask_ratio=mask_ratio,
        mask_ratio_adjust=mask_ratio_adjust,
        mask_block_size=mask_block_size,
        pad_id=pad_id,
        eos_id=eos_id,
    )

    bsz, num_views, seq_len = views.shape
    flat_views = views.reshape((bsz * num_views, seq_len))
    flat_mask = view_attn.reshape((bsz * num_views, seq_len))
    _reps_pre, reps_post, _pooled_post = model.forward_with_intermediates(
        flat_views,
        flat_mask,
        train=train,
        key=model_key,
    )
    token_post = reps_post.reshape((bsz, num_views, seq_len, reps_post.shape[-1]))
    return views, token_post, mask_positions, view_attn


def _masked_mean(
    reps: Array,
    mask: Array,
) -> Array:
    """Compute mean representations over masked positions.

    :param reps: Token representations of shape (B, V, T, D).
    :param mask: Boolean mask of shape (B, V, T).
    :returns: Mean pooled representations of shape (B, V, D).
    """
    if reps.ndim != 4:
        raise ValueError("reps must have shape (B, V, T, D)")
    if mask.ndim != 3:
        raise ValueError("mask must have shape (B, V, T)")
    if reps.shape[:3] != mask.shape:
        raise ValueError("reps and mask must align on (B, V, T)")

    mask_f = mask.astype(jnp.float32)
    reps_f = reps.astype(jnp.float32)
    summed = jnp.sum(reps_f * mask_f[..., None], axis=2)
    counts = jnp.sum(mask_f, axis=2)
    denom = jnp.maximum(counts, 1.0)
    return summed / denom[..., None]


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


def _save_checkpoint(
    run_dir: str,
    step: int,
    model: TextTransformer,
    teacher_params: eqx.Module,
    opt_state: eqx.Module,
    metadata: dict[str, object],
) -> None:
    """Save model, optimizer, and metadata to disk.

    :param run_dir: Run directory path.
    :param step: Global step index.
    :param model: Model to serialize.
    :param teacher_params: EMA teacher parameters to serialize.
    :param opt_state: Optimizer state to serialize.
    :param metadata: Metadata to persist.
    """
    ckpt_dir = os.path.join(run_dir, f"checkpoint_step_{step:08d}")
    os.makedirs(ckpt_dir, exist_ok=False)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "model.eqx"), model)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "swa.eqx"), teacher_params)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "optimizer.eqx"), opt_state)
    with open(os.path.join(ckpt_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    """Run the text training loop."""
    args = _parse_args()
    encoder_causal_attention = False
    predictor_causal_attention = False

    if args.max_train_steps < 0:
        raise ValueError("max-train-steps must be >= 0")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad-accum-steps must be > 0")
    if args.grad_clip_norm < 0.0:
        raise ValueError("grad-clip-norm must be >= 0")
    if args.log_every <= 0:
        raise ValueError("log-every must be > 0")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be > 0")
    if args.profile_warmup_steps < 0:
        raise ValueError("profile-warmup-steps must be >= 0")
    if args.num_views <= 0:
        raise ValueError("num-views must be > 0")
    if args.mask_ratio < 0.0 or args.mask_ratio > 1.0:
        raise ValueError("mask-ratio must be in [0, 1]")
    if args.mask_ratio_adjust < 0.0 or args.mask_ratio_adjust > 1.0:
        raise ValueError("mask-ratio-adjust must be in [0, 1]")
    if args.mask_block_size <= 0:
        raise ValueError("mask-block-size must be > 0")
    if args.data2vec_loss_weight < 0.0:
        raise ValueError("data2vec-loss-weight must be >= 0")
    if args.teacher_mode not in ("ema", "swa", "none"):
        raise ValueError("teacher-mode must be 'ema', 'swa', or 'none'")
    if args.teacher_mode == "ema":
        if args.teacher_ema_start < 0.0 or args.teacher_ema_start > 1.0:
            raise ValueError("teacher-ema-start must be in [0, 1]")
        if args.teacher_ema_end < 0.0 or args.teacher_ema_end > 1.0:
            raise ValueError("teacher-ema-end must be in [0, 1]")
        if args.teacher_ema_start > args.teacher_ema_end:
            raise ValueError("teacher-ema-start must be <= teacher-ema-end")
        if args.teacher_ema_steps < 0:
            raise ValueError("teacher-ema-steps must be >= 0")
    if args.teacher_top_k <= 0:
        raise ValueError("teacher-top-k must be > 0")
    if args.encoder_mlm_loss_weight < 0.0:
        raise ValueError("encoder-mlm-loss-weight must be >= 0")
    if args.encoder_mlm_keep_prob < 0.0 or args.encoder_mlm_keep_prob > 1.0:
        raise ValueError("encoder-mlm-keep-prob must be in [0, 1]")
    if args.sigreg_weight < 0.0 or args.sigreg_weight > 1.0:
        raise ValueError("sigreg-weight must be in [0, 1]")
    if args.sigreg_slices <= 0:
        raise ValueError("sigreg-slices must be > 0")
    if args.sigreg_seed < 0:
        raise ValueError("sigreg-seed must be >= 0")
    if args.data2vec_loss_weight + args.encoder_mlm_loss_weight <= 0.0:
        raise ValueError("at least one loss weight must be > 0")
    if args.data2vec_loss_weight > 0.0:
        if args.num_views <= 0:
            raise ValueError("data2vec requires at least one view")
        if args.mask_ratio <= 0.0:
            raise ValueError("data2vec requires mask-ratio > 0")

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
        _, views, view_masks, _view_attn = _build_views(
            jnp.asarray(batch_tokens),
            key,
            num_views=args.num_views,
            mask_ratio=args.mask_ratio,
            mask_ratio_adjust=args.mask_ratio_adjust,
            mask_block_size=args.mask_block_size,
            pad_id=pad_id,
            eos_id=eos_id,
        )

        for idx in range(batch_tokens.shape[0]):
            original = _decode_tokens(tokenizer, batch_tokens[idx], pad_id=pad_id)
            print(f"\nSample {idx + 1}")
            print("Original:")
            print(original)
            if args.num_views > 0:
                print("Masked views:")
                for view_idx in range(args.num_views):
                    view_tokens = np.asarray(views[idx, view_idx])
                    view_mask = np.asarray(view_masks[idx, view_idx])
                    view_tokens = np.where(view_mask, mask_id, view_tokens)
                    decoded = _decode_tokens(tokenizer, view_tokens, pad_id=pad_id)
                    print(f"  V{view_idx + 1}: {decoded}")
        dataset.close()
        return

    run_dir = _build_run_dir(args.runs_folder)

    predictor_layers = args.predictor_n_layers
    if predictor_layers is None:
        if args.data2vec_loss_weight > 0.0:
            predictor_layers = args.n_layers
        else:
            predictor_layers = 0
    if args.data2vec_loss_weight > 0.0 and predictor_layers <= 0:
        raise ValueError("predictor-n-layers must be > 0 when data2vec loss is enabled")
    if args.teacher_top_k > args.n_layers:
        raise ValueError("teacher-top-k must be <= n-layers")
    model_config = TextTransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        predictor_n_layers=int(predictor_layers),
        predictor_keep_unmasked=args.predictor_keep_unmasked,
        decoder_n_layers=0,
        mlp_ratio=args.mlp_ratio,
        attn_dropout=args.attn_dropout,
        resid_dropout=args.resid_dropout,
        drop_path_rate=args.drop_path_rate,
        pope_base=args.pope_base,
        init_std=args.init_std,
        attn_type=args.attn_type,
        embed_norm=False,
        embed_norm_scale=args.init_std,
        encoder_causal_attention=encoder_causal_attention,
        predictor_causal_attention=predictor_causal_attention,
        decoder_causal_attention=False,
    )
    exclusion_patterns = list(TextTransformer.MUON_PARAM_EXCLUSION_PATTERNS)
    weight_decay_exclusions = [
        r"^token_embed\..*$",
        r"^.*norm\d*\..*$",
        r"^.*norm.*$",
        r"^final_norm\..*$",
        r"^predictor\.final_norm\..*$",
        r"^predictor\.mask_tokens$",
    ]

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
            "num_views": args.num_views,
            "mask_ratio": args.mask_ratio,
            "mask_ratio_adjust": args.mask_ratio_adjust,
            "mask_block_size": args.mask_block_size,
            "predictor_keep_unmasked": args.predictor_keep_unmasked,
        },
        "loss": {
            "data2vec_loss_weight": args.data2vec_loss_weight,
            "teacher_mode": args.teacher_mode,
            "teacher_ema_start": args.teacher_ema_start,
            "teacher_ema_end": args.teacher_ema_end,
            "teacher_ema_steps": args.teacher_ema_steps,
            "teacher_top_k": args.teacher_top_k,
            "teacher_instance_norm": args.teacher_instance_norm,
            "encoder_mlm_weight": args.encoder_mlm_loss_weight,
            "encoder_mlm_keep_prob": args.encoder_mlm_keep_prob,
            "sigreg_weight": args.sigreg_weight,
            "sigreg_slices": args.sigreg_slices,
            "sigreg_seed": args.sigreg_seed,
            "sigreg_student": args.sigreg_student,
            "sigreg_mean_subtract": args.sigreg_mean_subtract,
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
            "grad_clip_norm": args.grad_clip_norm,
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
    weight_decay_mask = build_weight_decay_mask(params, flat_names, weight_decay_exclusions)

    flat_params, _ = jax.tree_util.tree_flatten(params)
    flat_muon, _ = jax.tree_util.tree_flatten(muon_mask)
    if len(flat_params) != len(flat_muon):
        raise ValueError("muon_mask must align with params")

    flat_wd, _ = jax.tree_util.tree_flatten(weight_decay_mask)
    if len(flat_wd) != len(flat_params):
        raise ValueError("weight_decay_mask must align with params")

    muon_names: list[str] = []
    adamw_names: list[str] = []
    for name, param, use_muon, use_wd in zip(flat_names, flat_params, flat_muon, flat_wd, strict=True):
        if isinstance(param, jax.Array) is False:
            continue
        suffix = " (weight decay on)" if use_wd is True else " (weight decay off)"
        if use_muon is True:
            muon_names.append(f"{name}{suffix}")
        else:
            adamw_names.append(f"{name}{suffix}")
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
        weight_decay_mask=weight_decay_mask,
    )

    opt_state: MuonWithAdamWFallbackState = optimizer.init(params)

    def _stop_gradient_tree(values: eqx.Module) -> eqx.Module:
        """Stop gradients for all array leaves in a pytree.

        :param values: Pytree of arrays and None entries.
        :returns: Pytree with gradients stopped for array leaves.
        """
        return jax.tree_util.tree_map(
            lambda leaf: jax.lax.stop_gradient(leaf) if isinstance(leaf, jax.Array) else leaf,
            values,
        )

    def _grad_step_impl(
        model_in: TextTransformer,
        teacher_params: eqx.Module,
        batch_tokens: Array,
        key: Array,
        global_step: Array,
        micro_step0: Array,
        micro_steps: Array,
        use_cond: bool,
    ) -> tuple[eqx.Module, Array, Array, Array, Array, Array, Array]:
        """Compute accumulated gradients and metrics across micro-steps.

        :param model_in: Replicated model.
        :param teacher_params: EMA teacher parameters used for target generation.
        :param batch_tokens: Token batches of shape (M, B, T).
        :param key: PRNG key for view masking.
        :param global_step: Global step index for loss scheduling.
        :param micro_step0: Starting micro-step index for RNG folding.
        :param micro_steps: Number of valid micro-steps in the batch.
        :param use_cond: Whether to guard micro-step evaluation with a conditional.
        :returns: Tuple of (grads, total loss, data2vec loss, sigreg loss,
            encoder mlm loss, encoder mlm acc1, encoder mlm acc5).
        """

        if args.teacher_mode == "none":
            teacher_model = model_in
        else:
            teacher_model = eqx.combine(_stop_gradient_tree(teacher_params), model_static)

        def _loss_fn(
            model_inner: TextTransformer,
            tokens: Array,
            tokens_key: Array,
        ) -> tuple[Array, tuple[Array, Array, Array, Array, Array]]:
            """Compute the total loss and its components.

            :param model_inner: Model replica used for the loss computation.
            :param tokens: Token batch of shape (B, T).
            :param tokens_key: PRNG key for masking.
            :returns: Tuple of (total loss, (data2vec loss, sigreg loss,
                encoder mlm loss, encoder mlm acc1, encoder mlm acc5)).
            """
            view_key, predictor_key, mlm_key = jax.random.split(tokens_key, 3)
            views, token_post, mask_positions, view_attn = _encode_views(
                model_inner,
                tokens,
                train=True,
                key=view_key,
                num_views=args.num_views,
                mask_ratio=args.mask_ratio,
                mask_ratio_adjust=args.mask_ratio_adjust,
                mask_block_size=args.mask_block_size,
                pad_id=pad_id,
                eos_id=eos_id,
            )

            bsz, num_views, seq_len, dim = token_post.shape
            total_views = args.num_views
            tokens_no_eos = jnp.where(tokens == eos_id, pad_id, tokens)
            encoder_mlm_loss = jnp.asarray(0.0, dtype=jnp.float32)
            encoder_mlm_acc1 = jnp.asarray(0.0, dtype=jnp.float32)
            encoder_mlm_acc5 = jnp.asarray(0.0, dtype=jnp.float32)
            if args.encoder_mlm_loss_weight > 0.0:
                if total_views > 0:
                    mlm_reps = token_post[:, :total_views, :, :]
                    mlm_masks = mask_positions[:, :total_views, :]
                    reps = mlm_reps.reshape((bsz * total_views, seq_len, dim))
                    mask = mlm_masks.reshape((bsz * total_views, seq_len))
                    targets = jnp.repeat(tokens_no_eos, repeats=total_views, axis=0)

                    if args.encoder_mlm_keep_prob > 0.0:
                        keep_key, mlm_key = jax.random.split(mlm_key)
                        keep_noise = jax.random.uniform(keep_key, shape=mask.shape)
                        keep_mask = keep_noise < args.encoder_mlm_keep_prob
                        keep_mask = jnp.logical_and(keep_mask, jnp.logical_not(mask))
                        loss_mask = jnp.logical_or(mask, keep_mask)
                    else:
                        keep_mask = jnp.zeros_like(mask)
                        loss_mask = mask

                    logits = model_inner.token_embed.unembed(reps)
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    target_logp = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
                    mask_f = loss_mask.astype(jnp.float32)
                    count = jnp.sum(mask_f)
                    loss_sum = -jnp.sum(target_logp * mask_f)
                    encoder_mlm_loss = jnp.where(count > 0.0, loss_sum / count, 0.0)

                    preds = jnp.argmax(logits, axis=-1)
                    correct1 = jnp.logical_and(preds == targets, mask)
                    acc1_sum = jnp.sum(correct1.astype(jnp.float32))
                    acc_count = jnp.sum(mask.astype(jnp.float32))
                    encoder_mlm_acc1 = jnp.where(acc_count > 0.0, acc1_sum / acc_count, 0.0)

                    top5 = jax.lax.top_k(logits, k=5)[1]
                    correct5 = jnp.any(top5 == targets[..., None], axis=-1)
                    correct5 = jnp.logical_and(correct5, mask)
                    acc5_sum = jnp.sum(correct5.astype(jnp.float32))
                    encoder_mlm_acc5 = jnp.where(acc_count > 0.0, acc5_sum / acc_count, 0.0)

            data2vec_loss = jnp.asarray(0.0, dtype=jnp.float32)
            sigreg_loss = jnp.asarray(0.0, dtype=jnp.float32)
            total_views = args.num_views
            if total_views > 0 and args.data2vec_loss_weight > 0.0:
                teacher_attn = tokens_no_eos != pad_id
                teacher_targets = _teacher_targets(
                    teacher_model,
                    tokens_no_eos,
                    teacher_attn,
                    top_k=args.teacher_top_k,
                    use_instance_norm=args.teacher_instance_norm,
                )
                if args.teacher_mode != "none":
                    teacher_targets = jax.lax.stop_gradient(teacher_targets)
                pred_in_reps = token_post[:, :total_views, :, :]
                pred_in_masks = mask_positions[:, :total_views, :]
                pred_in_attn = view_attn[:, :total_views, :]
                predictor_attn = jnp.logical_or(pred_in_attn, pred_in_masks)

                if args.sigreg_weight > 0.0:
                    sig_terms: list[Array] = []
                    if args.sigreg_student is True:
                        student_reps = pred_in_reps.astype(jnp.float32)
                        student_mask = pred_in_attn
                        student_mask_f = student_mask.astype(jnp.float32)
                        student_sum = jnp.sum(student_reps * student_mask_f[..., None], axis=2)
                        student_count = jnp.sum(student_mask_f, axis=2)
                        student_count = jnp.maximum(student_count, 1.0)
                        student_mean = student_sum / student_count[..., None]
                        if args.sigreg_mean_subtract is True:
                            student_points = student_reps - student_mean[:, :, None, :]
                        else:
                            student_points = student_reps

                        student_points = jnp.transpose(student_points, (0, 2, 1, 3))
                        student_mask_flat = jnp.transpose(student_mask, (0, 2, 1))
                        student_points = student_points.reshape((bsz * seq_len, total_views, dim))
                        student_mask_flat = student_mask_flat.reshape((bsz * seq_len, total_views))
                        student_sigreg = sigreg_loss_views_masked(
                            student_points,
                            student_mask_flat,
                            global_step=global_step,
                            num_slices=args.sigreg_slices,
                            seed=args.sigreg_seed,
                            axis_name="data",
                        )
                        sig_terms.append(student_sigreg)

                    if args.teacher_mode == "none":
                        teacher_mask_f = teacher_attn.astype(jnp.float32)
                        teacher_sum = jnp.sum(
                            teacher_targets * teacher_mask_f[:, :, None],
                            axis=1,
                            keepdims=True,
                        )
                        teacher_count = jnp.sum(teacher_mask_f, axis=1, keepdims=True)
                        teacher_count = jnp.maximum(teacher_count, 1.0)
                        teacher_mean = teacher_sum / teacher_count[:, :, None]
                        if args.sigreg_mean_subtract is True:
                            teacher_points = teacher_targets - teacher_mean
                        else:
                            teacher_points = teacher_targets

                        teacher_points = teacher_points.reshape((bsz * seq_len, 1, dim))
                        teacher_mask_flat = teacher_attn.reshape((bsz * seq_len, 1))
                        teacher_sigreg = sigreg_loss_views_masked(
                            teacher_points,
                            teacher_mask_flat,
                            global_step=global_step,
                            num_slices=args.sigreg_slices,
                            seed=args.sigreg_seed,
                            axis_name="data",
                        )
                        sig_terms.append(teacher_sigreg)

                    if len(sig_terms) > 0:
                        sigreg_loss = jnp.mean(jnp.stack(sig_terms, axis=0), axis=0)

                if model_inner.predictor is None:
                    raise ValueError("predictor must be enabled for data2vec loss")
                flat_tokens = pred_in_reps.reshape((bsz * total_views, seq_len, dim))
                flat_attn = predictor_attn.reshape((bsz * total_views, seq_len))
                flat_mask = pred_in_masks.reshape((bsz * total_views, seq_len))

                pred_reps = model_inner.predictor(
                    flat_tokens,
                    flat_attn,
                    flat_mask,
                    train=True,
                    key=predictor_key,
                )
                pred_reps = pred_reps.reshape((bsz, total_views, seq_len, dim))

                diffs = pred_reps.astype(jnp.float32) - teacher_targets[:, None, :, :]
                mse = jnp.mean(jnp.square(diffs), axis=-1)
                mask_f = pred_in_masks.astype(jnp.float32)
                loss_sum = jnp.sum(mse * mask_f)
                count = jnp.sum(mask_f)
                data2vec_loss = jnp.where(count > 0.0, loss_sum / count, 0.0)

                if args.sigreg_weight > 0.0:
                    sig_w = jnp.asarray(args.sigreg_weight, dtype=jnp.float32)
                    data2vec_loss = (1.0 - sig_w) * data2vec_loss + sig_w * sigreg_loss

            d2v_weight = jnp.asarray(args.data2vec_loss_weight, dtype=jnp.float32)
            mlm_weight = jnp.asarray(args.encoder_mlm_loss_weight, dtype=jnp.float32)
            weight_sum = jnp.maximum(d2v_weight + mlm_weight, 1e-6)
            d2v_frac = d2v_weight / weight_sum
            mlm_frac = mlm_weight / weight_sum
            total_loss = d2v_frac * data2vec_loss + mlm_frac * encoder_mlm_loss
            return (
                total_loss,
                (
                    data2vec_loss,
                    sigreg_loss,
                    encoder_mlm_loss,
                    encoder_mlm_acc1,
                    encoder_mlm_acc5,
                ),
            )

        value_and_grad = eqx.filter_value_and_grad(_loss_fn, has_aux=True)
        params_only = eqx.filter(model_in, eqx.is_array)
        grad_init = jax.tree_util.tree_map(
            lambda value: jnp.zeros_like(value) if value is not None else None,
            params_only,
        )
        loss_init = jnp.asarray(0.0, dtype=jnp.float32)
        data2vec_init = jnp.asarray(0.0, dtype=jnp.float32)
        sigreg_init = jnp.asarray(0.0, dtype=jnp.float32)
        encoder_mlm_init = jnp.asarray(0.0, dtype=jnp.float32)
        encoder_mlm_acc1_init = jnp.asarray(0.0, dtype=jnp.float32)
        encoder_mlm_acc5_init = jnp.asarray(0.0, dtype=jnp.float32)

        step_indices = jnp.arange(batch_tokens.shape[0], dtype=jnp.int32)

        def _compute_step(
            grads_acc: eqx.Module,
            loss_acc: Array,
            data2vec_acc: Array,
            sigreg_acc: Array,
            encoder_mlm_acc: Array,
            encoder_mlm_acc1_acc: Array,
            encoder_mlm_acc5_acc: Array,
            tokens: Array,
            tokens_key: Array,
        ) -> tuple[eqx.Module, Array, Array, Array, Array, Array]:
            """Evaluate gradients and accumulate metrics for one micro-step.

            :param grads_acc: Accumulated gradients so far.
            :param loss_acc: Accumulated total loss.
            :param data2vec_acc: Accumulated data2vec loss.
            :param sigreg_acc: Accumulated sigreg loss.
            :param encoder_mlm_acc: Accumulated encoder MLM loss.
            :param encoder_mlm_acc1_acc: Accumulated encoder MLM top-1 accuracy.
            :param encoder_mlm_acc5_acc: Accumulated encoder MLM top-5 accuracy.
            :param tokens: Token batch for this micro-step.
            :param tokens_key: PRNG key for this micro-step.
            :returns: Updated accumulators for gradients and metrics.
            """
            (
                loss,
                (
                    data2vec_loss,
                    sigreg_loss,
                    encoder_mlm_loss,
                    encoder_mlm_acc1,
                    encoder_mlm_acc5,
                ),
            ), grads = value_and_grad(
                model_in,
                tokens,
                tokens_key,
            )
            grads = _cast_tree_dtype(grads, jnp.float32)
            grads_accum = _add_trees(grads_acc, grads)
            loss_accum = loss_acc + loss
            data2vec_accum = data2vec_acc + data2vec_loss
            sigreg_accum = sigreg_acc + sigreg_loss
            encoder_mlm_accum = encoder_mlm_acc + encoder_mlm_loss
            encoder_mlm_acc1_accum = encoder_mlm_acc1_acc + encoder_mlm_acc1
            encoder_mlm_acc5_accum = encoder_mlm_acc5_acc + encoder_mlm_acc5
            return (
                grads_accum,
                loss_accum,
                data2vec_accum,
                sigreg_accum,
                encoder_mlm_accum,
                encoder_mlm_acc1_accum,
                encoder_mlm_acc5_accum,
            )

        if use_cond is True:
            def _accum_body(
                carry: tuple[eqx.Module, Array, Array, Array, Array, Array, Array],
                inputs: tuple[Array, Array],
            ) -> tuple[tuple[eqx.Module, Array, Array, Array, Array, Array, Array], None]:
                """Accumulate gradients and metrics for one micro-step.

                :param carry: Tuple of (grads, total loss, data2vec loss, sigreg loss,
                    encoder mlm loss, encoder mlm acc1, encoder mlm acc5).
                :param inputs: Tuple of (micro step index, token batch).
                :returns: Updated carry and unused output.
                """
                (
                    grads_acc,
                    loss_acc,
                    data2vec_acc,
                    sigreg_acc,
                    encoder_mlm_acc,
                    encoder_mlm_acc1_acc,
                    encoder_mlm_acc5_acc,
                ) = carry
                step_idx, tokens = inputs
                step_idx_global = micro_step0 + step_idx
                tokens_key = jax.random.fold_in(key, step_idx_global)

                def _do_compute(
                    _: None,
                ) -> tuple[eqx.Module, Array, Array, Array, Array, Array]:
                    return _compute_step(
                        grads_acc,
                        loss_acc,
                        data2vec_acc,
                        sigreg_acc,
                        encoder_mlm_acc,
                        encoder_mlm_acc1_acc,
                        encoder_mlm_acc5_acc,
                        tokens,
                        tokens_key,
                    )

                def _skip_compute(
                    _: None,
                ) -> tuple[eqx.Module, Array, Array, Array, Array, Array]:
                    return (
                        grads_acc,
                        loss_acc,
                        data2vec_acc,
                        sigreg_acc,
                        encoder_mlm_acc,
                        encoder_mlm_acc1_acc,
                        encoder_mlm_acc5_acc,
                    )

                active = step_idx < micro_steps
                new_carry = jax.lax.cond(active, _do_compute, _skip_compute, operand=None)
                return new_carry, None
        else:
            def _accum_body(
                carry: tuple[eqx.Module, Array, Array, Array, Array, Array, Array],
                inputs: tuple[Array, Array],
            ) -> tuple[tuple[eqx.Module, Array, Array, Array, Array, Array, Array], None]:
                """Accumulate gradients and metrics for one micro-step.

                :param carry: Tuple of (grads, total loss, data2vec loss, sigreg loss,
                    encoder mlm loss, encoder mlm acc1, encoder mlm acc5).
                :param inputs: Tuple of (micro step index, token batch).
                :returns: Updated carry and unused output.
                """
                (
                    grads_acc,
                    loss_acc,
                    data2vec_acc,
                    sigreg_acc,
                    encoder_mlm_acc,
                    encoder_mlm_acc1_acc,
                    encoder_mlm_acc5_acc,
                ) = carry
                step_idx, tokens = inputs
                step_idx_global = micro_step0 + step_idx
                tokens_key = jax.random.fold_in(key, step_idx_global)
                new_carry = _compute_step(
                    grads_acc,
                    loss_acc,
                    data2vec_acc,
                    sigreg_acc,
                    encoder_mlm_acc,
                    encoder_mlm_acc1_acc,
                    encoder_mlm_acc5_acc,
                    tokens,
                    tokens_key,
                )
                return new_carry, None

        (
            grads,
            loss,
            data2vec_loss,
            sigreg_loss,
            encoder_mlm_loss,
            encoder_mlm_acc1,
            encoder_mlm_acc5,
        ), _ = jax.lax.scan(
            _accum_body,
            (
                grad_init,
                loss_init,
                data2vec_init,
                sigreg_init,
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
        data2vec_loss = data2vec_loss / scale
        sigreg_loss = sigreg_loss / scale
        encoder_mlm_loss = encoder_mlm_loss / scale
        encoder_mlm_acc1 = encoder_mlm_acc1 / scale
        encoder_mlm_acc5 = encoder_mlm_acc5 / scale

        metrics = (
            loss,
            data2vec_loss,
            sigreg_loss,
            encoder_mlm_loss,
            encoder_mlm_acc1,
            encoder_mlm_acc5,
        )
        grads, metrics = jax.lax.pmean((grads, metrics), axis_name="data")
        (
            loss,
            data2vec_loss,
            sigreg_loss,
            encoder_mlm_loss,
            encoder_mlm_acc1,
            encoder_mlm_acc5,
        ) = metrics
        if args.grad_clip_norm > 0.0:
            grads, _grad_norm = _clip_grad_norm(grads, args.grad_clip_norm)
            _ = _grad_norm
        return (
            grads,
            loss,
            data2vec_loss,
            sigreg_loss,
            encoder_mlm_loss,
            encoder_mlm_acc1,
            encoder_mlm_acc5,
        )

    def grad_step(
        model_in: TextTransformer,
        teacher_params: eqx.Module,
        batch_tokens: Array,
        key: Array,
        global_step: Array,
        micro_step0: Array,
        micro_steps: Array,
    ) -> tuple[eqx.Module, Array, Array, Array, Array, Array, Array]:
        """Compute accumulated gradients with conditional micro-step guards.

        :param model_in: Replicated model.
        :param teacher_params: EMA teacher parameters used for target generation.
        :param batch_tokens: Token batches of shape (M, B, T).
        :param key: PRNG key for view masking.
        :param global_step: Global step index for loss scheduling.
        :param micro_step0: Starting micro-step index for RNG folding.
        :param micro_steps: Number of valid micro-steps in the batch.
        :returns: Tuple of (grads, total loss, data2vec loss, sigreg loss,
            encoder mlm loss, encoder mlm acc1, encoder mlm acc5).
        """
        return _grad_step_impl(
            model_in,
            teacher_params,
            batch_tokens,
            key,
            global_step,
            micro_step0,
            micro_steps,
            use_cond=True,
        )

    def grad_step_full(
        model_in: TextTransformer,
        teacher_params: eqx.Module,
        batch_tokens: Array,
        key: Array,
        global_step: Array,
        micro_step0: Array,
        micro_steps: Array,
    ) -> tuple[eqx.Module, Array, Array, Array, Array, Array, Array]:
        """Compute accumulated gradients when all micro-steps are active.

        :param model_in: Replicated model.
        :param teacher_params: EMA teacher parameters used for target generation.
        :param batch_tokens: Token batches of shape (M, B, T).
        :param key: PRNG key for view masking.
        :param global_step: Global step index for loss scheduling.
        :param micro_step0: Starting micro-step index for RNG folding.
        :param micro_steps: Number of valid micro-steps in the batch.
        :returns: Tuple of (grads, total loss, data2vec loss, sigreg loss,
            encoder mlm loss, encoder mlm acc1, encoder mlm acc5).
        """
        return _grad_step_impl(
            model_in,
            teacher_params,
            batch_tokens,
            key,
            global_step,
            micro_step0,
            micro_steps,
            use_cond=False,
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

    def _teacher_tau(step: Array) -> Array:
        """Compute EMA decay factor for the teacher at a given step.

        :param step: Global step index.
        :returns: EMA decay factor.
        """
        step_f = step.astype(jnp.float32)
        denom = jnp.asarray(args.teacher_ema_steps, dtype=jnp.float32)
        denom = jnp.maximum(denom, 1.0)
        progress = jnp.minimum(step_f / denom, 1.0)
        start = jnp.asarray(args.teacher_ema_start, dtype=jnp.float32)
        end = jnp.asarray(args.teacher_ema_end, dtype=jnp.float32)
        return start + (end - start) * progress

    def update_teacher_ema(
        teacher_params: eqx.Module,
        model_in: TextTransformer,
        step: Array,
    ) -> eqx.Module:
        """Update EMA teacher parameters with the latest student weights.

        :param teacher_params: Current teacher parameters.
        :param model_in: Current model (student).
        :param step: Global step index.
        :returns: Updated teacher parameters.
        """
        params_inner = eqx.filter(model_in, eqx.is_array)
        tau = _teacher_tau(step)

        def _update(teacher_value: Array | None, param: Array | None) -> Array | None:
            if param is None or teacher_value is None:
                return None
            return teacher_value * tau + param * (1.0 - tau)

        return jax.tree_util.tree_map(_update, teacher_params, params_inner)

    def update_teacher_swa(
        teacher_params: eqx.Module,
        model_in: TextTransformer,
        swa_count: Array,
    ) -> tuple[eqx.Module, Array]:
        """Update SWA teacher parameters with the latest student weights.

        :param teacher_params: Current teacher parameters.
        :param model_in: Current model (student).
        :param swa_count: Number of models averaged so far.
        :returns: Tuple of (updated SWA params, updated count).
        """
        params_inner = eqx.filter(model_in, eqx.is_array)
        count_f = swa_count.astype(jnp.float32)
        inv = 1.0 / (count_f + 1.0)

        def _update(teacher_value: Array | None, param: Array | None) -> Array | None:
            if param is None or teacher_value is None:
                return None
            return (teacher_value * count_f + param) * inv

        new_teacher = jax.tree_util.tree_map(_update, teacher_params, params_inner)
        return new_teacher, swa_count + 1

    grad_step_pmap = eqx.filter_pmap(
        grad_step,
        axis_name="data",
        devices=device_list,
    )  # type: ignore[call-overload]
    grad_step_full_pmap = eqx.filter_pmap(
        grad_step_full,
        axis_name="data",
        devices=device_list,
    )  # type: ignore[call-overload]
    apply_step_pmap = eqx.filter_pmap(
        apply_step,
        axis_name="data",
        devices=device_list,
        donate="all",
    )  # type: ignore[call-overload]
    update_teacher_ema_pmap = eqx.filter_pmap(
        update_teacher_ema,
        axis_name="data",
        devices=device_list,
    )  # type: ignore[call-overload]
    update_teacher_swa_pmap = eqx.filter_pmap(
        update_teacher_swa,
        axis_name="data",
        devices=device_list,
    )  # type: ignore[call-overload]

    params_repl = jax.device_put_replicated(params, device_list)
    train_repl = eqx.combine(params_repl, model_static)
    opt_state_repl = jax.device_put_replicated(opt_state, device_list)
    teacher_params_repl = jax.device_put_replicated(params, device_list)
    swa_count_repl: Array | None = None
    if args.teacher_mode == "swa":
        swa_count_repl = jax.device_put_replicated(
            jnp.asarray(1, dtype=jnp.int32),
            device_list,
        )

    global_batch = args.per_device_batch_size * num_devices
    global_step = 0
    perf_steps = 0
    perf_data_time = 0.0
    perf_compute_time = 0.0
    perf_log_time = 0.0
    perf_warmup = args.profile_warmup_steps
    last_loss_val = 0.0
    last_data2vec_val = 0.0
    last_sigreg_val = 0.0
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

                batch_tokens = jnp.stack(micro_batches, axis=1)
                batch_tokens = jax.device_put(batch_tokens, device=data_sharding)
                device_keys = jax.random.split(step_key, num_devices)
                device_keys = jax.device_put(device_keys, device=data_sharding)

                compute_start = time.perf_counter()
                if micro_steps == args.grad_accum_steps:
                    grad_fn = grad_step_full_pmap
                else:
                    grad_fn = grad_step_pmap
                (
                    grads,
                    loss,
                    data2vec_loss,
                    sigreg_loss,
                    encoder_mlm_loss,
                    encoder_mlm_acc1,
                    encoder_mlm_acc5,
                ) = grad_fn(
                    train_repl,
                    teacher_params_repl,
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
                if args.teacher_mode == "ema":
                    teacher_params_repl = update_teacher_ema_pmap(
                        teacher_params_repl,
                        train_repl,
                        step_id,
                    )
                elif args.teacher_mode == "swa":
                    if swa_count_repl is None:
                        raise ValueError("swa_count_repl must be set for SWA teacher updates")
                    teacher_params_repl, swa_count_repl = update_teacher_swa_pmap(
                        teacher_params_repl,
                        train_repl,
                        swa_count_repl,
                    )
                compute_time += time.perf_counter() - compute_start
                grads = None

                if args.profile:
                    jax.block_until_ready(loss)

                log_start = time.perf_counter()
                log_this_step = (global_step % args.log_every) == 0
                if log_this_step:
                    (
                        loss_host,
                        data2vec_host,
                        sigreg_host,
                        encoder_mlm_host,
                        encoder_mlm_acc1_host,
                        encoder_mlm_acc5_host,
                    ) = jax.device_get(
                        (
                            loss,
                            data2vec_loss,
                            sigreg_loss,
                            encoder_mlm_loss,
                            encoder_mlm_acc1,
                            encoder_mlm_acc5,
                        )
                    )
                    loss_val = float(np.mean(loss_host))
                    data2vec_val = float(np.mean(data2vec_host))
                    sigreg_val = float(np.mean(sigreg_host))
                    encoder_mlm_val = float(np.mean(encoder_mlm_host))
                    encoder_mlm_acc1_val = float(np.mean(encoder_mlm_acc1_host))
                    encoder_mlm_acc5_val = float(np.mean(encoder_mlm_acc5_host))
                    last_loss_val = loss_val
                    last_data2vec_val = data2vec_val
                    last_sigreg_val = sigreg_val
                    last_encoder_mlm_val = encoder_mlm_val
                    last_encoder_mlm_acc1_val = encoder_mlm_acc1_val
                    last_encoder_mlm_acc5_val = encoder_mlm_acc5_val

                    writer.add_scalar("train/total_loss", loss_val, global_step)
                    writer.add_scalar("train/data2vec_loss", data2vec_val, global_step)
                    writer.add_scalar("train/sigreg_loss", sigreg_val, global_step)
                    writer.add_scalar("train/encoder_mlm_loss", encoder_mlm_val, global_step)
                    writer.add_scalar("train/encoder_mlm_acc1", encoder_mlm_acc1_val, global_step)
                    writer.add_scalar("train/encoder_mlm_acc5", encoder_mlm_acc5_val, global_step)

                if log_this_step:
                    pbar.set_postfix(
                        total=f"{last_loss_val:.4f}",
                        d2v=f"{last_data2vec_val:.4f}",
                        sig=f"{last_sigreg_val:.4f}",
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
                    _block_until_ready(teacher_params_repl)
                    _block_until_ready(opt_state_repl)
                    opt_state_host = cast(MuonWithAdamWFallbackState, _unreplicate(opt_state_repl))
                    opt_state_host = cast(MuonWithAdamWFallbackState, _to_host(opt_state_host))
                    model_host = cast(TextTransformer, _unreplicate(train_repl))
                    model_host = cast(TextTransformer, _to_host(model_host))
                    teacher_host = cast(eqx.Module, _unreplicate(teacher_params_repl))
                    teacher_host = cast(eqx.Module, _to_host(teacher_host))
                    metadata = {
                        "global_step": global_step,
                        "config": run_config,
                    }
                    _save_checkpoint(
                        run_dir,
                        global_step,
                        model_host,
                        teacher_host,
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
