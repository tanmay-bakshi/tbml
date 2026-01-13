import argparse
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

from tbml.experiments.honeycomb.loss import lejepa_loss
from tbml.experiments.honeycomb.text.dataset import (
    StreamingTextDataset,
    _build_tokenizer,
    iter_text_batches,
)
from tbml.experiments.honeycomb.text.model import TextTransformer, TextTransformerConfig
from tbml.nn.init import truncated_normal_init
from tbml.nn.linear import Linear
from tbml.optimizers import MuonWithAdamWFallback, MuonWithAdamWFallbackState, build_muon_masks


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train a text encoder with LeJEPA.")
    parser.add_argument("--runs-folder", type=str, required=True)
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--eos-token", type=str, default="<|endoftext|>")
    parser.add_argument("--pad-token", type=str, default="<|pad|>")
    parser.add_argument("--mask-token", type=str, default="<|mask|>")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--full-sample-prob", type=float, default=1.0 / 3.0)
    parser.add_argument("--token-truncate-prob", type=float, default=1.0 / 3.0)
    parser.add_argument("--text-truncate-prob", type=float, default=1.0 / 3.0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup-steps", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=32)
    parser.add_argument("--num-devices", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--masked-probe", dest="masked_probe", action="store_true")
    parser.add_argument("--no-masked-probe", dest="masked_probe", action="store_false")
    parser.add_argument("--preview-views", action="store_true")

    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--resid-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--pope-base", type=float, default=10000.0)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--use-cls-token", dest="use_cls_token", action="store_true")
    parser.add_argument("--no-use-cls-token", dest="use_cls_token", action="store_false")

    parser.add_argument("--num-global-views", type=int, default=2)
    parser.add_argument("--num-local-views", type=int, default=6)
    parser.add_argument("--global-mask-min", type=float, default=None)
    parser.add_argument("--global-mask-max", type=float, default=None)
    parser.add_argument("--local-mask-min", type=float, default=None)
    parser.add_argument("--local-mask-max", type=float, default=None)
    parser.add_argument("--masking-mode", type=str, default="tokens", choices=["tokens", "spans"])

    parser.add_argument("--sigreg-weight", type=float, default=0.25)
    parser.add_argument("--sigreg-slices", type=int, default=256)
    parser.add_argument("--sigreg-seed", type=int, default=0)
    parser.add_argument("--pred-loss", type=str, default="mse", choices=["mse", "cosine"])

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
        use_cls_token=False,
        masked_probe=False,
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


def _list_jsonl_files(folder: str) -> list[str]:
    """Collect JSONL file paths from a folder.

    :param folder: Directory containing JSONL files.
    :returns: Sorted list of JSONL file paths.
    """
    if os.path.isdir(folder) is False:
        raise FileNotFoundError(f"data folder not found: {folder}")
    paths: list[str] = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".jsonl") is False:
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path) is False:
            continue
        paths.append(path)
    if len(paths) == 0:
        raise FileNotFoundError(f"no .jsonl files found in folder: {folder}")
    return paths


def _count_text_samples(folder: str, text_field: str) -> int:
    """Count valid text samples across JSONL files.

    :param folder: Directory containing JSONL files.
    :param text_field: JSON field name containing the text.
    :returns: Number of valid samples.
    """
    files = _list_jsonl_files(folder)
    total = 0
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get(text_field)
                if isinstance(text, str):
                    total += 1
    return total


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


def _prefix_patterns(patterns: list[str], prefix: str) -> list[str]:
    """Prefix regex patterns with a dotted module path.

    :param patterns: Regex patterns to prefix.
    :param prefix: Prefix to insert after the anchor.
    :returns: Prefixed regex patterns.
    """
    prefixed: list[str] = []
    for pattern in patterns:
        if pattern.startswith("^"):
            prefixed.append(f"^{prefix}\\.{pattern[1:]}")
        else:
            prefixed.append(f"^{prefix}\\.{pattern}")
    return prefixed


class MaskedTokenProbe(eqx.Module):
    """Linear probe for masked-token prediction.

    :ivar vocab_size: Vocabulary size for logits.
    :ivar d_model: Input representation dimension.
    :ivar dtype: Compute dtype.
    :ivar param_dtype: Parameter dtype.
    :ivar proj: Linear projection to vocab logits.
    """

    vocab_size: int
    d_model: int
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    proj: Linear

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        *,
        init_std: float,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        key: Array,
    ) -> None:
        """Initialize the masked-token probe.

        :param d_model: Input representation dimension.
        :param vocab_size: Vocabulary size.
        :param init_std: Truncated normal standard deviation.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param key: PRNG key for parameter initialization.
        """
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if init_std <= 0.0:
            raise ValueError("init_std must be > 0")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dtype = dtype
        self.param_dtype = param_dtype

        init = truncated_normal_init(init_std)
        self.proj = Linear(
            in_features=d_model,
            out_features=vocab_size,
            use_bias=True,
            bias_value=0.0,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            key=key,
        )

    def __call__(self, reps: Array) -> Array:
        """Project token representations to vocabulary logits.

        :param reps: Token representations of shape (..., d_model).
        :returns: Logits of shape (..., vocab_size).
        """
        if reps.shape[-1] != self.d_model:
            raise ValueError("reps last dimension must match d_model")
        return self.proj(reps)


class TextTrainBundle(eqx.Module):
    """Container for the model and optional probe.

    :ivar model: Text transformer model.
    :ivar probe: Optional masked-token probe.
    """

    model: TextTransformer
    probe: MaskedTokenProbe | None


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
    iterator: Iterable[tuple[np.ndarray, np.ndarray]],
    size: int,
    sharding: Sharding,
) -> Iterator[tuple[Array, Array]]:
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
        yield cast(tuple[Array, Array], payload)


def _iter_batches(
    dataset: StreamingTextDataset,
    *,
    batch_size: int,
    max_seq_len: int,
    pad_id: int,
    num_devices: int,
    per_device_batch: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield device-sharded batches from the streaming dataset.

    :param dataset: Streaming dataset instance.
    :param batch_size: Global batch size.
    :param max_seq_len: Maximum sequence length.
    :param pad_id: Padding token id.
    :param num_devices: Number of devices.
    :param per_device_batch: Batch size per device.
    :returns: Iterable of sharded (tokens, attention_mask) batches.
    """
    host_iter = iter_text_batches(
        dataset,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        pad_id=pad_id,
    )
    for tokens, mask in host_iter:
        tokens = tokens.reshape((num_devices, per_device_batch, max_seq_len))
        mask = mask.reshape((num_devices, per_device_batch, max_seq_len))
        yield tokens, mask


def _mask_tokens(
    tokens: Array,
    key: Array,
    *,
    min_ratio: float,
    max_ratio: float,
    mask_id: int,
    pad_id: int,
    eos_id: int,
) -> tuple[Array, Array, Array]:
    """Apply random masking to tokens.

    :param tokens: Token ids of shape (B, T).
    :param key: PRNG key.
    :param min_ratio: Minimum masking ratio.
    :param max_ratio: Maximum masking ratio.
    :param mask_id: Masking token id.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :returns: Tuple of (masked token ids, mask positions).
    """
    if min_ratio < 0.0 or min_ratio > 1.0:
        raise ValueError("min_ratio must be in [0, 1]")
    if max_ratio < 0.0 or max_ratio > 1.0:
        raise ValueError("max_ratio must be in [0, 1]")
    if min_ratio > max_ratio:
        raise ValueError("min_ratio must be <= max_ratio")

    key_ratio, key_mask = jax.random.split(key)
    ratios = jax.random.uniform(
        key_ratio,
        shape=(tokens.shape[0],),
        minval=min_ratio,
        maxval=max_ratio,
    )
    rand = jax.random.uniform(key_mask, shape=tokens.shape, minval=0.0, maxval=1.0)
    eligible = jnp.logical_and(tokens != pad_id, tokens != eos_id)
    mask = jnp.logical_and(rand < ratios[:, None], eligible)
    masked_tokens = jnp.where(mask, jnp.asarray(mask_id, dtype=tokens.dtype), tokens)
    return masked_tokens, mask


def _mask_spans(
    tokens: Array,
    key: Array,
    *,
    min_ratio: float,
    max_ratio: float,
    mask_id: int,
    pad_id: int,
    eos_id: int,
    poisson_lambda: float = 3.0,
) -> tuple[Array, Array, Array]:
    """Apply span masking to tokens using BART-style text infilling.

    Spans are replaced by a single mask token and the sequence is compacted.

    :param tokens: Token ids of shape (B, T).
    :param key: PRNG key.
    :param min_ratio: Minimum masking ratio.
    :param max_ratio: Maximum masking ratio.
    :param mask_id: Masking token id.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :param poisson_lambda: Poisson lambda for span lengths.
    :returns: Tuple of (masked token ids, mask positions, attention mask).
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
        span_lengths = jax.random.poisson(key_len, poisson_lambda, shape=(seq_len,))
        span_lengths = jnp.maximum(span_lengths, 1)
        start_scores = jax.random.uniform(key_start, shape=(seq_len, seq_len), minval=0.0, maxval=1.0)

        def _body(i: int, state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            mask, start_mask, masked_count = state
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
            start_positions = idx == start
            apply = jnp.logical_and(use, any_possible)
            new_mask = jnp.where(apply, jnp.logical_or(mask, span_positions), mask)
            new_start = jnp.where(apply, jnp.logical_or(start_mask, start_positions), start_mask)
            new_count = jnp.where(apply, masked_count + span_len, masked_count)
            return new_mask, new_start, new_count

        init_mask = jnp.zeros((seq_len,), dtype=jnp.bool_)
        init_start = jnp.zeros((seq_len,), dtype=jnp.bool_)
        init_count = jnp.asarray(0, dtype=jnp.int32)
        final_mask, final_start, _final_count = jax.lax.fori_loop(
            0, seq_len, _body, (init_mask, init_start, init_count)
        )

        def _compact(
            idx: int,
            state: tuple[Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array]:
            output_tokens, output_positions, write_idx, last_was_mask = state
            keep = jnp.logical_or(final_start[idx], jnp.logical_not(final_mask[idx]))
            keep = jnp.logical_and(keep, sample_tokens[idx] != pad_id)
            is_mask = final_start[idx]
            should_write = jnp.logical_and(
                keep,
                jnp.logical_or(jnp.logical_not(is_mask), jnp.logical_not(last_was_mask)),
            )
            value = jnp.where(is_mask, mask_id, sample_tokens[idx])
            output_tokens = jnp.where(
                should_write,
                output_tokens.at[write_idx].set(value),
                output_tokens,
            )
            output_positions = jnp.where(
                should_write,
                output_positions.at[write_idx].set(is_mask),
                output_positions,
            )
            write_idx = jnp.where(should_write, write_idx + 1, write_idx)
            last_was_mask = jnp.where(should_write, is_mask, last_was_mask)
            return output_tokens, output_positions, write_idx, last_was_mask

        output_tokens = jnp.full((seq_len,), pad_id, dtype=sample_tokens.dtype)
        output_positions = jnp.zeros((seq_len,), dtype=jnp.bool_)
        output_tokens, output_positions, final_len, _last_was_mask = jax.lax.fori_loop(
            0, seq_len, _compact, (output_tokens, output_positions, 0, jnp.asarray(False))
        )
        output_mask = jnp.arange(seq_len, dtype=jnp.int32) < final_len
        return output_tokens, output_positions, output_mask

    masked, positions, attn_mask = jax.vmap(_one_sample)(tokens, ratios, keys)
    return masked, positions, attn_mask


def _mask_views(
    tokens: Array,
    key: Array,
    *,
    num_views: int,
    min_ratio: float,
    max_ratio: float,
    masking_mode: str,
    mask_id: int,
    pad_id: int,
    eos_id: int,
) -> tuple[Array, Array, Array]:
    """Generate multiple masked views of the token batch.

    :param tokens: Token ids of shape (B, T).
    :param key: PRNG key.
    :param num_views: Number of views to generate.
    :param min_ratio: Minimum masking ratio.
    :param max_ratio: Maximum masking ratio.
    :param masking_mode: Masking mode ("tokens" or "spans").
    :param mask_id: Masking token id.
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
        :returns: Tuple of (masked token ids, mask positions).
        """
        if masking_mode == "tokens":
            masked_tokens, mask_positions = _mask_tokens(
                tokens,
                view_key,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                mask_id=mask_id,
                pad_id=pad_id,
                eos_id=eos_id,
            )
            attn_mask = tokens != pad_id
            return masked_tokens, mask_positions, attn_mask
        if masking_mode == "spans":
            return _mask_spans(
                tokens,
                view_key,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                mask_id=mask_id,
                pad_id=pad_id,
                eos_id=eos_id,
            )
        raise ValueError("masking_mode must be 'tokens' or 'spans'")

    views, masks, attn_masks = jax.vmap(_one_view)(keys)
    return (
        jnp.transpose(views, (1, 0, 2)),
        jnp.transpose(masks, (1, 0, 2)),
        jnp.transpose(attn_masks, (1, 0, 2)),
    )


def _pool_representations(
    model: TextTransformer,
    reps: Array,
    attention_mask: Array,
) -> Array:
    """Pool token representations and apply final normalization.

    :param model: Text transformer model.
    :param reps: Token representations of shape (B, T, d_model).
    :param attention_mask: Attention mask of shape (B, T).
    :returns: Pooled embeddings of shape (B, d_model).
    """
    if reps.ndim != 3:
        raise ValueError("reps must have shape (B, T, d_model)")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape (B, T)")
    if reps.shape[:2] != attention_mask.shape:
        raise ValueError("attention_mask must match reps batch and sequence dimensions")

    mask = attention_mask.astype(reps.dtype)
    lengths = jnp.sum(mask, axis=1)
    if model.config.use_cls_token is True:
        idx = jnp.maximum(lengths.astype(jnp.int32) - 1, 0)
        idx = idx[:, None, None]
        idx = jnp.broadcast_to(idx, (idx.shape[0], 1, reps.shape[-1]))
        pooled = jnp.take_along_axis(reps, idx, axis=1).squeeze(axis=1)
    else:
        masked = reps * mask[:, :, None]
        denom = jnp.maximum(lengths, 1.0)
        pooled = jnp.sum(masked, axis=1) / denom[:, None]
    return model.final_norm(pooled)


def _encode_views(
    model: TextTransformer,
    tokens: Array,
    attention_mask: Array,
    *,
    train: bool,
    key: Array,
    num_global_views: int,
    num_local_views: int,
    global_mask_min: float,
    global_mask_max: float,
    local_mask_min: float,
    local_mask_max: float,
    masking_mode: str,
    mask_id: int,
    pad_id: int,
    eos_id: int,
    return_tokens: bool = False,
) -> tuple[Array, Array | None, Array | None]:
    """Encode masked views into pooled embeddings.

    :param model: Text transformer encoder.
    :param tokens: Token ids of shape (B, T).
    :param attention_mask: Attention mask of shape (B, T).
    :param train: Whether to enable dropout.
    :param key: PRNG key.
    :param num_global_views: Number of global views.
    :param num_local_views: Number of local views.
    :param global_mask_min: Minimum global masking ratio.
    :param global_mask_max: Maximum global masking ratio.
    :param local_mask_min: Minimum local masking ratio.
    :param local_mask_max: Maximum local masking ratio.
    :param masking_mode: Masking mode ("tokens" or "spans").
    :param mask_id: Masking token id.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :param return_tokens: Whether to return token representations and masks.
    :returns: Tuple of (pooled embeddings, token reps, mask positions).
    """
    if num_global_views + num_local_views <= 0:
        raise ValueError("at least one view must be requested")

    key, global_key, local_key = jax.random.split(key, 3)
    global_views, global_masks, global_attn = _mask_views(
        tokens,
        global_key,
        num_views=num_global_views,
        min_ratio=global_mask_min,
        max_ratio=global_mask_max,
        masking_mode=masking_mode,
        mask_id=mask_id,
        pad_id=pad_id,
        eos_id=eos_id,
    )
    local_views, local_masks, local_attn = _mask_views(
        tokens,
        local_key,
        num_views=num_local_views,
        min_ratio=local_mask_min,
        max_ratio=local_mask_max,
        masking_mode=masking_mode,
        mask_id=mask_id,
        pad_id=pad_id,
        eos_id=eos_id,
    )
    if num_local_views > 0 and num_global_views > 0:
        views = jnp.concatenate([global_views, local_views], axis=1)
        mask_positions = jnp.concatenate([global_masks, local_masks], axis=1)
        view_attn = jnp.concatenate([global_attn, local_attn], axis=1)
    elif num_global_views > 0:
        views = global_views
        mask_positions = global_masks
        view_attn = global_attn
    else:
        views = local_views
        mask_positions = local_masks
        view_attn = local_attn

    bsz, num_views, seq_len = views.shape
    flat_views = views.reshape((bsz * num_views, seq_len))
    flat_mask = view_attn.reshape((bsz * num_views, seq_len))
    if return_tokens is True:
        token_reps = model.encode_tokens(flat_views, flat_mask, train=train, key=key)
        pooled = _pool_representations(model, token_reps, flat_mask)
        pooled = pooled.reshape((bsz, num_views, pooled.shape[-1]))
        token_reps = token_reps.reshape((bsz, num_views, seq_len, token_reps.shape[-1]))
        return pooled, token_reps, mask_positions
    pooled = model(flat_views, flat_mask, train=train, key=key)
    return pooled.reshape((bsz, num_views, pooled.shape[-1])), None, None


def _masked_probe_metrics(
    probe: MaskedTokenProbe,
    token_reps: Array,
    mask_positions: Array,
    tokens: Array,
) -> tuple[Array, Array]:
    """Compute masked-token probe loss and accuracy.

    :param probe: Linear probe mapping to vocabulary logits.
    :param token_reps: Token representations of shape (B, V, T, d_model).
    :param mask_positions: Mask positions of shape (B, V, T).
    :param tokens: Original token ids of shape (B, T).
    :returns: Tuple of (cross entropy loss, accuracy).
    """
    if token_reps.ndim != 4:
        raise ValueError("token_reps must have shape (B, V, T, d_model)")
    if mask_positions.ndim != 3:
        raise ValueError("mask_positions must have shape (B, V, T)")
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (B, T)")

    reps = jax.lax.stop_gradient(token_reps)
    targets = jnp.broadcast_to(tokens[:, None, :], mask_positions.shape)
    logits = probe(reps)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    nll = -target_log_probs
    mask = mask_positions.astype(nll.dtype)
    masked_nll = nll * mask
    total_masked = jnp.sum(mask)
    loss = jnp.where(total_masked > 0, jnp.sum(masked_nll) / total_masked, 0.0)

    preds = jnp.argmax(logits, axis=-1)
    correct = preds == targets
    correct_masked = jnp.sum(jnp.where(mask_positions, correct, False))
    acc = jnp.where(
        total_masked > 0,
        correct_masked.astype(jnp.float32) / total_masked,
        0.0,
    )
    return loss.astype(jnp.float32), acc


def _save_checkpoint(
    run_dir: str,
    step: int,
    model: TextTransformer,
    opt_state: eqx.Module,
    probe: MaskedTokenProbe | None,
    metadata: dict[str, object],
) -> None:
    """Save model, optimizer, and metadata to disk.

    :param run_dir: Run directory path.
    :param step: Global step index.
    :param model: Model to serialize.
    :param opt_state: Optimizer state to serialize.
    :param probe: Optional masked-token probe module.
    :param metadata: Metadata to persist.
    """
    ckpt_dir = os.path.join(run_dir, f"checkpoint_step_{step:08d}")
    os.makedirs(ckpt_dir, exist_ok=False)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "model.eqx"), model)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "optimizer.eqx"), opt_state)
    if probe is not None:
        eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "probe.eqx"), probe)
    with open(os.path.join(ckpt_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    """Run the text training loop."""
    args = _parse_args()

    if args.max_train_steps < 0:
        raise ValueError("max-train-steps must be >= 0")
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
    if args.masking_mode == "spans":
        default_global_min = 0.25
        default_global_max = 0.30
        default_local_min = 0.40
        default_local_max = 0.45
    else:
        default_global_min = 0.20
        default_global_max = 0.40
        default_local_min = 0.60
        default_local_max = 0.70

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
    if args.full_sample_prob < 0.0 or args.token_truncate_prob < 0.0 or args.text_truncate_prob < 0.0:
        raise ValueError("sample probabilities must be >= 0")
    if args.full_sample_prob + args.token_truncate_prob + args.text_truncate_prob <= 0.0:
        raise ValueError("sum of sample probabilities must be > 0")
    if args.masked_probe is True and args.masking_mode == "spans":
        raise ValueError("masked-probe is only supported with masking-mode 'tokens'")

    dtype = _dtype_from_name(args.dtype)
    betas = _parse_betas(args.adamw_betas)
    total_samples = _count_text_samples(args.data_folder, args.text_field)

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

    dataset = StreamingTextDataset(
        args.data_folder,
        text_field=args.text_field,
        tokenizer_name=args.tokenizer,
        eos_token=args.eos_token,
        pad_token=args.pad_token,
        mask_token=args.mask_token,
        max_seq_len=args.max_seq_len,
        shuffle_buffer=args.shuffle_buffer,
        full_sample_prob=args.full_sample_prob,
        token_truncate_prob=args.token_truncate_prob,
        text_truncate_prob=args.text_truncate_prob,
        num_workers=args.num_workers,
        prefetch=args.prefetch,
        seed=args.seed,
    )
    vocab_size, eos_id, pad_id, mask_id = dataset.tokenizer_info()

    if args.preview_views is True:
        tokenizer, _eos_id, _pad_id, _mask_id = _build_tokenizer(
            args.tokenizer,
            eos_token=args.eos_token,
            pad_token=args.pad_token,
            mask_token=args.mask_token,
        )
        global_batch = args.per_device_batch_size * num_devices
        preview_iter = iter_text_batches(
            dataset,
            batch_size=global_batch,
            max_seq_len=args.max_seq_len,
            pad_id=pad_id,
        )
        batch_tokens, batch_mask = next(iter(preview_iter))
        del batch_mask
        key = jax.random.PRNGKey(args.seed)
        key, global_key, local_key = jax.random.split(key, 3)
        global_views, _global_masks, _global_attn = _mask_views(
            jnp.asarray(batch_tokens),
            global_key,
            num_views=args.num_global_views,
            min_ratio=args.global_mask_min,
            max_ratio=args.global_mask_max,
            masking_mode=args.masking_mode,
            mask_id=mask_id,
            pad_id=pad_id,
            eos_id=eos_id,
        )
        local_views, _local_masks, _local_attn = _mask_views(
            jnp.asarray(batch_tokens),
            local_key,
            num_views=args.num_local_views,
            min_ratio=args.local_mask_min,
            max_ratio=args.local_mask_max,
            masking_mode=args.masking_mode,
            mask_id=mask_id,
            pad_id=pad_id,
            eos_id=eos_id,
        )

        print(f"Masking mode: {args.masking_mode}")
        for idx in range(batch_tokens.shape[0]):
            original = _decode_tokens(tokenizer, batch_tokens[idx], pad_id=pad_id)
            print(f"\nSample {idx + 1}")
            print("Original:")
            print(original)
            if args.num_global_views > 0:
                print("Global views:")
                for view_idx in range(args.num_global_views):
                    view_tokens = np.asarray(global_views[idx, view_idx])
                    decoded = _decode_tokens(tokenizer, view_tokens, pad_id=pad_id)
                    print(f"  G{view_idx + 1}: {decoded}")
            if args.num_local_views > 0:
                print("Local views:")
                for view_idx in range(args.num_local_views):
                    view_tokens = np.asarray(local_views[idx, view_idx])
                    decoded = _decode_tokens(tokenizer, view_tokens, pad_id=pad_id)
                    print(f"  L{view_idx + 1}: {decoded}")
        dataset.close()
        return

    run_dir = _build_run_dir(args.runs_folder)

    model_config = TextTransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        attn_dropout=args.attn_dropout,
        resid_dropout=args.resid_dropout,
        drop_path_rate=args.drop_path_rate,
        pope_base=args.pope_base,
        init_std=args.init_std,
        use_cls_token=args.use_cls_token,
    )
    if args.masked_probe is True:
        exclusion_patterns = _prefix_patterns(
            list(TextTransformer.MUON_PARAM_EXCLUSION_PATTERNS), "model"
        )
        exclusion_patterns.append(r"^probe\..*$")
    else:
        exclusion_patterns = list(TextTransformer.MUON_PARAM_EXCLUSION_PATTERNS)

    run_config: dict[str, object] = {
        "model": model_config.model_dump(),
        "data": {
            "text_field": args.text_field,
            "tokenizer": args.tokenizer,
            "eos_token": args.eos_token,
            "pad_token": args.pad_token,
            "mask_token": args.mask_token,
            "max_seq_len": args.max_seq_len,
            "shuffle_buffer": args.shuffle_buffer,
            "full_sample_prob": args.full_sample_prob,
            "token_truncate_prob": args.token_truncate_prob,
            "text_truncate_prob": args.text_truncate_prob,
            "vocab_size": vocab_size,
            "eos_id": eos_id,
            "pad_id": pad_id,
            "mask_id": mask_id,
        },
        "views": {
            "num_global_views": args.num_global_views,
            "num_local_views": args.num_local_views,
            "global_mask_min": args.global_mask_min,
            "global_mask_max": args.global_mask_max,
            "local_mask_min": args.local_mask_min,
            "local_mask_max": args.local_mask_max,
            "masking_mode": args.masking_mode,
        },
        "loss": {
            "sigreg_weight": args.sigreg_weight,
            "sigreg_slices": args.sigreg_slices,
            "sigreg_seed": args.sigreg_seed,
            "pred_loss": args.pred_loss,
        },
        "probe": {
            "masked_probe": args.masked_probe,
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
            "per_device_batch_size": args.per_device_batch_size,
            "num_devices": num_devices,
            "seed": args.seed,
            "dtype": args.dtype,
            "log_every": args.log_every,
            "checkpoint_every": args.checkpoint_every,
            "profile": args.profile,
            "profile_warmup_steps": args.profile_warmup_steps,
            "num_workers": args.num_workers,
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
        param_dtype=dtype,
        key=model_key,
    )
    probe: MaskedTokenProbe | None = None
    train_bundle: TextTrainBundle | None = None
    if args.masked_probe is True:
        probe_key, base_key = jax.random.split(base_key)
        probe = MaskedTokenProbe(
            d_model=model_config.d_model,
            vocab_size=vocab_size,
            init_std=model_config.init_std,
            dtype=dtype,
            param_dtype=dtype,
            key=probe_key,
        )
        train_bundle = TextTrainBundle(model=model, probe=probe)

    if args.masked_probe is True:
        if train_bundle is None:
            raise ValueError("train_bundle must be initialized when masked_probe is enabled")
        params, static = eqx.partition(train_bundle, eqx.is_array)
    else:
        params, static = eqx.partition(model, eqx.is_array)
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

    if args.masked_probe is True:
        def train_step(
            bundle_in: TextTrainBundle,
            state_in: MuonWithAdamWFallbackState,
            batch_tokens: Array,
            batch_mask: Array,
            key: Array,
            global_step: Array,
        ) -> tuple[
            TextTrainBundle,
            MuonWithAdamWFallbackState,
            Array,
            Array,
            Array,
            Array,
            Array,
        ]:
            """Run one training step with the masked-token probe.

            :param bundle_in: Replicated model and probe bundle.
            :param state_in: Replicated optimizer state.
            :param batch_tokens: Token batch of shape (B, T).
            :param batch_mask: Attention mask of shape (B, T).
            :param key: PRNG key for view masking.
            :param global_step: Global step index for loss scheduling.
            :returns: Updated bundle, optimizer state, and metrics.
            """

            def _loss_fn(
                bundle_inner: TextTrainBundle,
            ) -> tuple[Array, tuple[Array, Array, Array, Array, Array]]:
                """Compute LeJEPA losses and probe metrics.

                :param bundle_inner: Bundle containing model and probe.
                :returns: Tuple of (total loss, (model loss, pred, sigreg, probe loss, probe acc)).
                """
                model_inner = bundle_inner.model
                if bundle_inner.probe is None:
                    raise ValueError("probe must be present when masked_probe is enabled")
                emb, token_reps, mask_positions = _encode_views(
                    model_inner,
                    batch_tokens,
                    batch_mask,
                    train=True,
                    key=key,
                    num_global_views=args.num_global_views,
                    num_local_views=args.num_local_views,
                    global_mask_min=args.global_mask_min,
                    global_mask_max=args.global_mask_max,
                    local_mask_min=args.local_mask_min,
                    local_mask_max=args.local_mask_max,
                    masking_mode=args.masking_mode,
                    mask_id=mask_id,
                    pad_id=pad_id,
                    eos_id=eos_id,
                    return_tokens=True,
                )
                if token_reps is None or mask_positions is None:
                    raise ValueError("token_reps and mask_positions must be returned for probe training")
                model_loss, pred, sigreg = lejepa_loss(
                    emb,
                    args.num_global_views,
                    sigreg_weight=args.sigreg_weight,
                    pred_loss_type=args.pred_loss,
                    global_step=global_step,
                    num_slices=args.sigreg_slices,
                    seed=args.sigreg_seed,
                    axis_name="data",
                )
                probe_loss, probe_acc = _masked_probe_metrics(
                    bundle_inner.probe,
                    token_reps,
                    mask_positions,
                    batch_tokens,
                )
                total = model_loss + probe_loss
                return total, (model_loss, pred, sigreg, probe_loss, probe_acc)

            (
                total_loss,
                (loss, pred_loss, sigreg_loss, probe_loss, probe_acc),
            ), grads = eqx.filter_value_and_grad(
                _loss_fn, has_aux=True
            )(bundle_in)
            grads = jax.lax.pmean(grads, axis_name="data")
            params_inner = eqx.filter(bundle_in, eqx.is_array)
            updates, new_state = optimizer.update(grads, state_in, params_inner)
            new_bundle = eqx.apply_updates(bundle_in, updates)
            _ = total_loss
            loss = jax.lax.pmean(loss, axis_name="data")
            pred_loss = jax.lax.pmean(pred_loss, axis_name="data")
            sigreg_loss = jax.lax.pmean(sigreg_loss, axis_name="data")
            probe_loss = jax.lax.pmean(probe_loss, axis_name="data")
            probe_acc = jax.lax.pmean(probe_acc, axis_name="data")
            return (
                new_bundle,
                new_state,
                loss,
                pred_loss,
                sigreg_loss,
                probe_loss,
                probe_acc,
            )

        train_step_pmap = eqx.filter_pmap(
            train_step,
            axis_name="data",
            devices=device_list,
            donate="all",
        )  # type: ignore[call-overload]
    else:
        def train_step(
            model_in: TextTransformer,
            state_in: MuonWithAdamWFallbackState,
            batch_tokens: Array,
            batch_mask: Array,
            key: Array,
            global_step: Array,
        ) -> tuple[TextTransformer, MuonWithAdamWFallbackState, Array, Array, Array]:
            """Run one training step.

            :param model_in: Replicated model.
            :param state_in: Replicated optimizer state.
            :param batch_tokens: Token batch of shape (B, T).
            :param batch_mask: Attention mask of shape (B, T).
            :param key: PRNG key for view masking.
            :param global_step: Global step index for loss scheduling.
            :returns: Updated model, optimizer state, and losses.
            """

            def _loss_fn(model_inner: TextTransformer) -> tuple[Array, tuple[Array, Array]]:
                """Compute the total loss and its components.

                :param model_inner: Model replica used for the loss computation.
                :returns: Tuple of (total loss, (prediction loss, sigreg loss)).
                """
                emb, _token_reps, _mask_positions = _encode_views(
                    model_inner,
                    batch_tokens,
                    batch_mask,
                    train=True,
                    key=key,
                    num_global_views=args.num_global_views,
                    num_local_views=args.num_local_views,
                    global_mask_min=args.global_mask_min,
                    global_mask_max=args.global_mask_max,
                    local_mask_min=args.local_mask_min,
                    local_mask_max=args.local_mask_max,
                    masking_mode=args.masking_mode,
                    mask_id=mask_id,
                    pad_id=pad_id,
                    eos_id=eos_id,
                    return_tokens=False,
                )
                total, pred, sigreg = lejepa_loss(
                    emb,
                    args.num_global_views,
                    sigreg_weight=args.sigreg_weight,
                    pred_loss_type=args.pred_loss,
                    global_step=global_step,
                    num_slices=args.sigreg_slices,
                    seed=args.sigreg_seed,
                    axis_name="data",
                )
                return total, (pred, sigreg)

            (loss, (pred_loss, sigreg_loss)), grads = eqx.filter_value_and_grad(
                _loss_fn, has_aux=True
            )(model_in)
            grads = jax.lax.pmean(grads, axis_name="data")
            params_inner = eqx.filter(model_in, eqx.is_array)
            updates, new_state = optimizer.update(grads, state_in, params_inner)
            new_model = eqx.apply_updates(model_in, updates)
            loss = jax.lax.pmean(loss, axis_name="data")
            pred_loss = jax.lax.pmean(pred_loss, axis_name="data")
            sigreg_loss = jax.lax.pmean(sigreg_loss, axis_name="data")
            return new_model, new_state, loss, pred_loss, sigreg_loss

        train_step_pmap = eqx.filter_pmap(
            train_step,
            axis_name="data",
            devices=device_list,
            donate="all",
        )  # type: ignore[call-overload]

    if args.masked_probe is True:
        params_repl = _replicate_tree(params, num_devices)
        params_repl = jax.device_put(params_repl, device=data_sharding)
        train_repl = eqx.combine(params_repl, static)
    else:
        model_params, model_static = params, static
        model_params_repl = _replicate_tree(model_params, num_devices)
        model_params_repl = jax.device_put(model_params_repl, device=data_sharding)
        train_repl = eqx.combine(model_params_repl, model_static)
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
    last_pred_val = 0.0
    last_sig_val = 0.0
    last_probe_loss_val = 0.0
    last_probe_acc_val = 0.0
    if total_samples <= 0:
        raise ValueError("no valid samples found in the dataset")
    epoch_steps = total_samples // global_batch
    if epoch_steps <= 0:
        raise ValueError("dataset too small for the configured batch size")
    if args.max_train_steps > 0:
        total_steps = min(args.max_train_steps, epoch_steps)
    else:
        total_steps = epoch_steps

    try:
        train_iter_host = _iter_batches(
            dataset,
            batch_size=global_batch,
            max_seq_len=args.max_seq_len,
            pad_id=pad_id,
            num_devices=num_devices,
            per_device_batch=args.per_device_batch_size,
        )
        train_iter = _prefetch_to_device(
            train_iter_host,
            size=args.prefetch,
            sharding=data_sharding,
        )
        train_iter = iter(train_iter)

        with tqdm(total=total_steps, desc="Train") as pbar:
            for _ in range(total_steps):
                step_start = time.perf_counter()
                batch_tokens, batch_mask = next(train_iter)
                data_done = time.perf_counter()
                step_key = jax.random.fold_in(base_key, global_step)
                device_keys = jax.random.split(step_key, num_devices)
                device_keys = jax.device_put(device_keys, device=data_sharding)
                step_id = jnp.full((num_devices,), global_step, dtype=jnp.int32)

                if args.masked_probe is True:
                    (
                        train_repl,
                        opt_state_repl,
                        loss,
                        pred,
                        sigreg,
                        probe_loss,
                        probe_acc,
                    ) = train_step_pmap(
                        train_repl,
                        opt_state_repl,
                        batch_tokens,
                        batch_mask,
                        device_keys,
                        step_id,
                    )
                else:
                    train_repl, opt_state_repl, loss, pred, sigreg = train_step_pmap(
                        train_repl,
                        opt_state_repl,
                        batch_tokens,
                        batch_mask,
                        device_keys,
                        step_id,
                    )
                if args.profile:
                    jax.block_until_ready(loss)
                compute_done = time.perf_counter()

                log_this_step = (global_step % args.log_every) == 0
                if log_this_step:
                    loss_val = float(np.mean(jax.device_get(loss)))
                    pred_val = float(np.mean(jax.device_get(pred)))
                    sig_val = float(np.mean(jax.device_get(sigreg)))
                    last_loss_val = loss_val
                    last_pred_val = pred_val
                    last_sig_val = sig_val

                    writer.add_scalar("train/total_loss", loss_val, global_step)
                    writer.add_scalar("train/pred_loss", pred_val, global_step)
                    writer.add_scalar("train/sigreg_loss", sig_val, global_step)
                if args.masked_probe is True:
                    probe_loss_val = float(np.mean(jax.device_get(probe_loss)))
                    probe_acc_val = float(np.mean(jax.device_get(probe_acc)))
                    last_probe_loss_val = probe_loss_val
                    last_probe_acc_val = probe_acc_val
                    writer.add_scalar("probe/cross_entropy", probe_loss_val, global_step)
                    writer.add_scalar("probe/accuracy", probe_acc_val, global_step)

                if log_this_step or args.masked_probe is True:
                    if args.masked_probe is True:
                        pbar.set_postfix(
                            total=f"{last_loss_val:.4f}",
                            pred=f"{last_pred_val:.4f}",
                            sigreg=f"{last_sig_val:.4f}",
                            probe_loss=f"{last_probe_loss_val:.4f}",
                            probe_acc=f"{last_probe_acc_val:.4f}",
                        )
                    else:
                        pbar.set_postfix(
                            total=f"{last_loss_val:.4f}",
                            pred=f"{last_pred_val:.4f}",
                            sigreg=f"{last_sig_val:.4f}",
                        )
                pbar.update(1)
                global_step += 1
                log_done = time.perf_counter()

                if args.profile:
                    if perf_steps + perf_warmup < global_step:
                        perf_steps += 1
                        perf_data_time += data_done - step_start
                        perf_compute_time += compute_done - data_done
                        perf_log_time += log_done - compute_done

                if global_step % args.checkpoint_every == 0:
                    jax.block_until_ready(loss)
                    opt_state_host = cast(MuonWithAdamWFallbackState, _unreplicate(opt_state_repl))
                    probe_host: MaskedTokenProbe | None = None
                    if args.masked_probe is True:
                        bundle_host = cast(TextTrainBundle, _unreplicate(train_repl))
                        bundle_host = cast(TextTrainBundle, _to_host(bundle_host))
                        model_host = bundle_host.model
                        probe_host = bundle_host.probe
                        if probe_host is None:
                            raise ValueError("probe missing from train bundle for checkpointing")
                    else:
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
                        probe_host,
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
