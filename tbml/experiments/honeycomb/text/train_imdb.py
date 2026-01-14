import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime
import traceback
from typing import BinaryIO, Iterable, Iterator, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from jaxtyping import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding
from tensorboardX import SummaryWriter  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from tbml.experiments.honeycomb.text.dataset import _build_tokenizer
from tbml.experiments.honeycomb.text.model import TextTransformer, TextTransformerConfig
from tbml.nn.init import truncated_normal_init
from tbml.nn.linear import Linear
from tbml.optimizers import MuonWithAdamWFallback, MuonWithAdamWFallbackState, build_muon_masks


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a text model on IMDB sentiment.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--runs-folder", type=str, required=True)
    parser.add_argument("--data-folder", type=str, default="misc/imdb")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--label-field", type=str, default="label")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--train-backbone-after", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup-steps", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=32)
    parser.add_argument("--num-devices", type=int, default=0)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--shuffle", dest="shuffle", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")

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
        shuffle=True,
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


def _load_run_config(checkpoint_dir: str) -> dict[str, object]:
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


def _skip_deserialise(_handle: BinaryIO, leaf: object) -> object:
    """Skip deserialising a leaf and keep the default value.

    :param _handle: Binary file handle (unused).
    :param leaf: Leaf value from the target tree.
    :returns: The unchanged leaf value.
    """
    return leaf


def _is_d_out_path(path: jax.tree_util.KeyPath) -> bool:
    """Check whether a PyTree path points to a SwiGLU d_out field.

    :param path: PyTree path entries.
    :returns: True if the path ends with ``d_out``.
    """
    if len(path) == 0:
        return False
    last = path[-1]
    return isinstance(last, jax.tree_util.GetAttrKey) and last.name == "d_out"


def _build_legacy_filter_spec(model: eqx.Module) -> object:
    """Build a deserialisation filter spec for legacy checkpoints.

    :param model: Model instance used as the target tree.
    :returns: Filter spec PyTree.
    """
    paths_and_leaves, tree_def = jax.tree_util.tree_flatten_with_path(model)
    flat_leaves, _ = jax.tree_util.tree_flatten(model)
    if len(paths_and_leaves) != len(flat_leaves):
        raise ValueError("tree flatten mismatch while building legacy filter spec")
    filter_leaves: list[object] = []
    for (path, _leaf), leaf in zip(paths_and_leaves, flat_leaves, strict=True):
        if _is_d_out_path(path):
            filter_leaves.append(_skip_deserialise)
        else:
            filter_leaves.append(eqx.default_deserialise_filter_spec)
    return jax.tree_util.tree_unflatten(tree_def, filter_leaves)


def _load_with_dtype_fallback(
    model_path: str,
    model: TextTransformer,
) -> TextTransformer:
    """Load a model checkpoint, retrying on dtype mismatches.

    :param model_path: Path to the model checkpoint.
    :param model: Model instance for shape/dtype reference.
    :returns: Loaded model instance.
    """
    candidates = [jnp.float32, jnp.bfloat16, jnp.float16]
    seen: set[jnp.dtype] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        model_key = jax.random.PRNGKey(0)
        config = model.config
        model_candidate = TextTransformer(
            config,
            dtype=candidate,
            param_dtype=candidate,
            key=model_key,
        )
        try:
            return eqx.tree_deserialise_leaves(model_path, model_candidate)
        except RuntimeError as exc:
            last_error = exc
            if "changed dtype" not in str(exc):
                raise
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("failed to load model checkpoint")


def _load_checkpoint_model(
    model_path: str,
    model: TextTransformer,
    *,
    patch_dir: str,
) -> TextTransformer:
    """Load a model checkpoint with legacy compatibility handling.

    :param model_path: Path to the model checkpoint.
    :param model: Model instance for shape/dtype reference.
    :param patch_dir: Directory to write patched checkpoints into.
    :returns: Loaded model instance.
    """
    try:
        return _load_with_dtype_fallback(model_path, model)
    except (RuntimeError, ValueError) as exc:
        print("Detected legacy checkpoint format; patching d_out fields.")
        filter_spec = _build_legacy_filter_spec(model)
        try:
            patched = eqx.tree_deserialise_leaves(
                model_path,
                model,
                filter_spec=filter_spec,
            )
        except Exception as inner_exc:
            stack = traceback.format_exc()
            raise RuntimeError(
                f"Failed to load legacy checkpoint: {inner_exc}\\n{stack}"
            ) from exc
        patched_path = os.path.join(patch_dir, "model_legacy_patched.eqx")
        eqx.tree_serialise_leaves(patched_path, patched)
        return patched


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


def _devices_for_platform(platform: str) -> list[jax.Device]:
    """List JAX devices matching a platform name.

    :param platform: Platform string such as "cpu" or "gpu".
    :returns: List of matching devices.
    """
    return [device for device in jax.devices() if device.platform == platform]


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


def _prefetch_to_device(
    iterator: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    size: int,
    sharding: Sharding,
) -> Iterator[tuple[Array, Array, Array]]:
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
        yield cast(tuple[Array, Array, Array], payload)


def _collect_parquet_paths(folder: str, split: str) -> list[str]:
    """Collect parquet files for the given split.

    :param folder: Directory containing parquet files.
    :param split: Dataset split name.
    :returns: Sorted list of parquet paths.
    """
    if os.path.isdir(folder) is False:
        raise FileNotFoundError(f"data folder not found: {folder}")
    paths: list[str] = []
    prefix = f"{split}-"
    for name in sorted(os.listdir(folder)):
        if name.startswith(prefix) is False:
            continue
        if name.endswith(".parquet") is False:
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path) is False:
            continue
        paths.append(path)
    if len(paths) == 0:
        raise FileNotFoundError(f"no parquet files found for split: {split}")
    return paths


def _load_imdb_table(
    folder: str,
    *,
    split: str,
    text_field: str,
    label_field: str,
) -> tuple[list[str], np.ndarray]:
    """Load IMDB samples from parquet files.

    :param folder: Dataset folder path.
    :param split: Dataset split name.
    :param text_field: Column name for text.
    :param label_field: Column name for labels.
    :returns: Tuple of (texts, labels).
    """
    paths = _collect_parquet_paths(folder, split)
    tables = [pq.read_table(path, columns=[text_field, label_field]) for path in paths]
    table = pa.concat_tables(tables)
    texts_raw = table[text_field].to_pylist()
    labels_raw = table[label_field].to_numpy()
    texts: list[str] = []
    labels_list: list[int] = []
    for text_val, label_val in zip(texts_raw, labels_raw, strict=True):
        if isinstance(text_val, str) is False:
            continue
        texts.append(text_val)
        labels_list.append(int(label_val))
    labels = np.asarray(labels_list, dtype=np.int64)
    if len(texts) == 0:
        raise ValueError("no valid text samples found in dataset")
    if labels.shape[0] != len(texts):
        raise ValueError("labels and texts length mismatch")
    return texts, labels


class ImdbDataset:
    """In-memory IMDB dataset for sentiment classification."""

    _tokens: list[list[int]]
    _labels: np.ndarray
    _pad_id: int
    _max_seq_len: int
    _seed: int
    _shuffle: bool

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        *,
        tokenizer_name: str,
        eos_token: str,
        pad_token: str,
        mask_token: str,
        max_seq_len: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        """Initialize the IMDB dataset.

        :param texts: List of input texts.
        :param labels: Array of integer labels.
        :param tokenizer_name: Hugging Face tokenizer identifier.
        :param eos_token: EOS token string.
        :param pad_token: Padding token string.
        :param mask_token: Masking token string.
        :param max_seq_len: Maximum sequence length.
        :param seed: Random seed for shuffling.
        :param shuffle: Whether to shuffle each pass.
        """
        if max_seq_len <= 1:
            raise ValueError("max_seq_len must be > 1")
        if labels.shape[0] != len(texts):
            raise ValueError("labels and texts length mismatch")

        tokenizer, eos_id, pad_id, _mask_id = _build_tokenizer(
            tokenizer_name,
            eos_token=eos_token,
            pad_token=pad_token,
            mask_token=mask_token,
        )

        tokens: list[list[int]] = []
        for text in texts:
            encoding = tokenizer.encode(text, add_special_tokens=False)
            token_ids = list(encoding.ids)
            if len(token_ids) > max_seq_len - 1:
                token_ids = token_ids[: max_seq_len - 1]
            token_ids.append(int(eos_id))
            tokens.append(token_ids)

        self._tokens = tokens
        self._labels = labels.astype(np.int64)
        self._pad_id = int(pad_id)
        self._max_seq_len = max_seq_len
        self._seed = seed
        self._shuffle = shuffle

    def __len__(self) -> int:
        """Return number of samples.

        :returns: Number of samples in the dataset.
        """
        return len(self._tokens)

    def iter_batches(self, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Iterate over padded batches endlessly.

        :param batch_size: Batch size per host.
        :returns: Iterator yielding (tokens, attention_mask, labels).
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        rng = np.random.default_rng(self._seed)
        num_samples = len(self._tokens)
        if num_samples <= 0:
            raise ValueError("dataset is empty")

        indices = np.arange(num_samples)
        while True:
            if self._shuffle is True:
                rng.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                if batch_idx.shape[0] < batch_size:
                    continue
                tokens = np.full((batch_size, self._max_seq_len), self._pad_id, dtype=np.int32)
                attention_mask = np.zeros((batch_size, self._max_seq_len), dtype=np.bool_)
                labels = np.empty((batch_size,), dtype=np.int32)
                for row, idx in enumerate(batch_idx):
                    sample = self._tokens[int(idx)]
                    length = min(len(sample), self._max_seq_len)
                    if length > 0:
                        tokens[row, :length] = np.asarray(sample[:length], dtype=np.int32)
                        attention_mask[row, :length] = True
                    labels[row] = int(self._labels[int(idx)])
                yield tokens, attention_mask, labels


def _iter_batches(
    iterator: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    batch_size: int,
    max_seq_len: int,
    num_devices: int,
    per_device_batch: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield device-sharded batches from a host iterator.

    :param iterator: Host iterator yielding (tokens, mask, labels).
    :param batch_size: Global batch size.
    :param max_seq_len: Maximum sequence length.
    :param num_devices: Number of devices.
    :param per_device_batch: Batch size per device.
    :returns: Iterable of sharded (tokens, attention_mask, labels) batches.
    """
    for tokens, mask, labels in iterator:
        tokens = tokens.reshape((num_devices, per_device_batch, max_seq_len))
        mask = mask.reshape((num_devices, per_device_batch, max_seq_len))
        labels = labels.reshape((num_devices, per_device_batch))
        yield tokens, mask, labels


def _iter_epoch_batches(
    dataset: ImdbDataset,
    *,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Iterate over a single epoch of batches.

    :param dataset: IMDB dataset instance.
    :param batch_size: Batch size per host.
    :param rng: Random generator for shuffling.
    :param shuffle: Whether to shuffle the dataset order.
    :returns: Iterator yielding (tokens, attention_mask, labels).
    """
    num_samples = len(dataset._tokens)
    if num_samples <= 0:
        raise ValueError("dataset is empty")
    indices = np.arange(num_samples)
    if shuffle is True:
        rng.shuffle(indices)
    for start in range(0, num_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        if batch_idx.shape[0] < batch_size:
            continue
        tokens = np.full((batch_size, dataset._max_seq_len), dataset._pad_id, dtype=np.int32)
        attention_mask = np.zeros((batch_size, dataset._max_seq_len), dtype=np.bool_)
        labels = np.empty((batch_size,), dtype=np.int32)
        for row, idx in enumerate(batch_idx):
            sample = dataset._tokens[int(idx)]
            length = min(len(sample), dataset._max_seq_len)
            if length > 0:
                tokens[row, :length] = np.asarray(sample[:length], dtype=np.int32)
                attention_mask[row, :length] = True
            labels[row] = int(dataset._labels[int(idx)])
        yield tokens, attention_mask, labels


class SentimentProbe(eqx.Module):
    """Linear classifier for sentiment.

    :ivar d_model: Input representation dimension.
    :ivar num_classes: Number of output classes.
    :ivar proj: Linear projection to logits.
    """

    d_model: int
    num_classes: int
    proj: Linear

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        *,
        init_std: float,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        key: Array,
    ) -> None:
        """Initialize the classifier.

        :param d_model: Input representation dimension.
        :param num_classes: Number of output classes.
        :param init_std: Truncated normal standard deviation.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param key: PRNG key for parameter initialization.
        """
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1")
        if init_std <= 0.0:
            raise ValueError("init_std must be > 0")

        self.d_model = d_model
        self.num_classes = num_classes
        init = truncated_normal_init(init_std)
        self.proj = Linear(
            in_features=d_model,
            out_features=num_classes,
            use_bias=True,
            bias_value=0.0,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            key=key,
        )

    def __call__(self, reps: Array) -> Array:
        """Project pooled representations to logits.

        :param reps: Pooled representations of shape (B, d_model).
        :returns: Logits of shape (B, num_classes).
        """
        if reps.shape[-1] != self.d_model:
            raise ValueError("reps last dimension must match d_model")
        return self.proj(reps)


class SentimentBundle(eqx.Module):
    """Bundle of model and sentiment probe.

    :ivar model: Text transformer model.
    :ivar classifier: Sentiment classifier head.
    """

    model: TextTransformer
    classifier: SentimentProbe


def _save_checkpoint(
    run_dir: str,
    epoch: int,
    bundle: SentimentBundle,
    opt_state: eqx.Module,
    metadata: dict[str, object],
) -> None:
    """Save model, classifier, optimizer, and metadata to disk.

    :param run_dir: Run directory path.
    :param epoch: Epoch index.
    :param bundle: Training bundle to serialize.
    :param opt_state: Optimizer state to serialize.
    :param metadata: Metadata to persist.
    """
    ckpt_dir = os.path.join(run_dir, f"checkpoint_epoch_{epoch:04d}")
    os.makedirs(ckpt_dir, exist_ok=False)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "bundle.eqx"), bundle)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "optimizer.eqx"), opt_state)
    with open(os.path.join(ckpt_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    """Run IMDB fine-tuning."""
    args = _parse_args()

    if args.max_train_steps < 0:
        raise ValueError("max-train-steps must be >= 0")
    if args.max_val_steps < 0:
        raise ValueError("max-val-steps must be >= 0")
    if args.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if args.log_every <= 0:
        raise ValueError("log-every must be > 0")
    if args.profile_warmup_steps < 0:
        raise ValueError("profile-warmup-steps must be >= 0")
    if args.train_backbone_after is not None and args.train_backbone_after < 0:
        raise ValueError("train-backbone-after must be >= 0 when set")

    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint)
    base_config = _load_run_config(checkpoint_dir)
    model_config_raw = base_config.get("model")
    if model_config_raw is None or isinstance(model_config_raw, dict) is False:
        raise ValueError("checkpoint missing model configuration")
    data_config_raw = base_config.get("data")
    if data_config_raw is None or isinstance(data_config_raw, dict) is False:
        raise ValueError("checkpoint missing data configuration")
    training_config_raw = base_config.get("training")
    if training_config_raw is None or isinstance(training_config_raw, dict) is False:
        raise ValueError("checkpoint missing training configuration")

    if args.from_scratch is True:
        dtype = _dtype_from_name(args.dtype)
    else:
        checkpoint_dtype = training_config_raw.get("dtype")
        if checkpoint_dtype is None or isinstance(checkpoint_dtype, str) is False:
            raise ValueError("checkpoint training config missing dtype")
        dtype = _dtype_from_name(checkpoint_dtype)
    dtype_name = jnp.dtype(dtype).name

    betas = _parse_betas(args.adamw_betas)
    run_dir = _build_run_dir(args.runs_folder)

    tokenizer_name = data_config_raw.get("tokenizer")
    eos_token = data_config_raw.get("eos_token")
    pad_token = data_config_raw.get("pad_token")
    mask_token = data_config_raw.get("mask_token")
    max_seq_len = data_config_raw.get("max_seq_len")
    if tokenizer_name is None or isinstance(tokenizer_name, str) is False:
        raise ValueError("checkpoint data config missing tokenizer")
    if eos_token is None or isinstance(eos_token, str) is False:
        raise ValueError("checkpoint data config missing eos token")
    if pad_token is None or isinstance(pad_token, str) is False:
        raise ValueError("checkpoint data config missing pad token")
    if mask_token is None or isinstance(mask_token, str) is False:
        raise ValueError("checkpoint data config missing mask token")
    if max_seq_len is None or isinstance(max_seq_len, int) is False:
        raise ValueError("checkpoint data config missing max_seq_len")

    model_config = TextTransformerConfig(**model_config_raw)

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

    train_texts, train_labels = _load_imdb_table(
        args.data_folder,
        split="train",
        text_field=args.text_field,
        label_field=args.label_field,
    )
    val_texts, val_labels = _load_imdb_table(
        args.data_folder,
        split="test",
        text_field=args.text_field,
        label_field=args.label_field,
    )
    train_dataset = ImdbDataset(
        train_texts,
        train_labels,
        tokenizer_name=tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
        max_seq_len=max_seq_len,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    val_dataset = ImdbDataset(
        val_texts,
        val_labels,
        tokenizer_name=tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
        max_seq_len=max_seq_len,
        seed=args.seed,
        shuffle=args.shuffle,
    )

    base_key = jax.random.PRNGKey(args.seed)
    model_key, base_key = jax.random.split(base_key)
    model = TextTransformer(
        model_config,
        dtype=dtype,
        param_dtype=dtype,
        key=model_key,
    )
    if args.from_scratch is False:
        print("Training from checkpoint.")
        model_path = os.path.join(checkpoint_dir, "model.eqx")
        model = _load_checkpoint_model(model_path, model, patch_dir=run_dir)
    else:
        print("Training from scratch.")

    probe_key, base_key = jax.random.split(base_key)
    classifier = SentimentProbe(
        d_model=model_config.d_model,
        num_classes=2,
        init_std=model_config.init_std,
        dtype=dtype,
        param_dtype=dtype,
        key=probe_key,
    )
    bundle = SentimentBundle(model=model, classifier=classifier)

    exclusion_patterns = _prefix_patterns(
        list(TextTransformer.MUON_PARAM_EXCLUSION_PATTERNS), "model"
    )
    exclusion_patterns.append(r"^classifier\..*$")

    params, static = eqx.partition(bundle, eqx.is_array)
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

    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    trainable_flags: list[bool] = []
    for name, param in zip(flat_names, flat_params, strict=True):
        trainable_flags.append(isinstance(param, jax.Array) and name.startswith("classifier."))
    trainable_mask = jax.tree_util.tree_unflatten(tree_def, trainable_flags)

    run_config: dict[str, object] = {
        "model": model_config.model_dump(),
        "data": {
            "folder": args.data_folder,
            "train_split": "train",
            "val_split": "test",
            "text_field": args.text_field,
            "label_field": args.label_field,
            "tokenizer": tokenizer_name,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "max_seq_len": max_seq_len,
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
            "from_scratch": args.from_scratch,
            "train_backbone_after": args.train_backbone_after,
            "epochs": args.epochs,
            "max_train_steps": args.max_train_steps,
            "max_val_steps": args.max_val_steps,
            "per_device_batch_size": args.per_device_batch_size,
            "num_devices": num_devices,
            "seed": args.seed,
            "dtype": dtype_name,
            "log_every": args.log_every,
            "profile": args.profile,
            "profile_warmup_steps": args.profile_warmup_steps,
            "shuffle": args.shuffle,
            "prefetch": args.prefetch,
        },
    }

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    writer = SummaryWriter(run_dir)
    writer.add_text("config", json.dumps(run_config, indent=2))

    def train_step(
        bundle_in: SentimentBundle,
        state_in: MuonWithAdamWFallbackState,
        batch_tokens: Array,
        batch_mask: Array,
        labels_in: Array,
        key: Array,
        train_backbone: bool,
    ) -> tuple[SentimentBundle, MuonWithAdamWFallbackState, Array, Array]:
        """Run one fine-tuning step.

        :param bundle_in: Replicated training bundle.
        :param state_in: Replicated optimizer state.
        :param batch_tokens: Token batch of shape (B, T).
        :param batch_mask: Attention mask of shape (B, T).
        :param labels_in: Label ids of shape (B,).
        :param key: PRNG key for dropout.
        :param train_backbone: Whether to update the backbone this step.
        :returns: Updated bundle, optimizer state, loss, and accuracy.
        """

        def _loss_fn(bundle_inner: SentimentBundle) -> tuple[Array, Array]:
            """Compute cross-entropy loss and accuracy.

            :param bundle_inner: Bundle containing model and classifier.
            :returns: Tuple of (loss, accuracy).
            """
            model_inner = bundle_inner.model
            classifier_inner = bundle_inner.classifier
            pooled = model_inner(batch_tokens, batch_mask, train=train_backbone, key=key)
            if train_backbone is False:
                pooled = jax.lax.stop_gradient(pooled)
            logits = classifier_inner(pooled)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            labels_b = labels_in.astype(jnp.int32)
            nll = -jnp.take_along_axis(log_probs, labels_b[:, None], axis=-1).squeeze(-1)
            loss = jnp.mean(nll)
            preds = jnp.argmax(logits, axis=-1)
            acc = jnp.mean(preds == labels_b)
            return loss, acc

        (loss, acc), grads = eqx.filter_value_and_grad(_loss_fn, has_aux=True)(bundle_in)
        grads = jax.lax.pmean(grads, axis_name="data")
        params_inner = eqx.filter(bundle_in, eqx.is_array)
        updates, new_state = optimizer.update(grads, state_in, params_inner)
        if train_backbone is False:
            updates = jax.tree_util.tree_map(
                lambda update, flag: update if flag else None,
                updates,
                trainable_mask,
            )
        new_bundle = eqx.apply_updates(bundle_in, updates)
        loss = jax.lax.pmean(loss, axis_name="data")
        acc = jax.lax.pmean(acc, axis_name="data")
        return new_bundle, new_state, loss, acc

    train_step_pmap = eqx.filter_pmap(
        train_step,
        axis_name="data",
        devices=device_list,
        donate="all",
    )  # type: ignore[call-overload]

    def val_step(
        bundle_in: SentimentBundle,
        batch_tokens: Array,
        batch_mask: Array,
        labels_in: Array,
    ) -> tuple[Array, Array]:
        """Run one validation step.

        :param bundle_in: Replicated training bundle.
        :param batch_tokens: Token batch of shape (B, T).
        :param batch_mask: Attention mask of shape (B, T).
        :param labels_in: Label ids of shape (B,).
        :returns: Loss and accuracy.
        """
        pooled = bundle_in.model(batch_tokens, batch_mask, train=False, key=None)
        logits = bundle_in.classifier(pooled)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        labels_b = labels_in.astype(jnp.int32)
        nll = -jnp.take_along_axis(log_probs, labels_b[:, None], axis=-1).squeeze(-1)
        loss = jnp.mean(nll)
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels_b)
        loss = jax.lax.pmean(loss, axis_name="data")
        acc = jax.lax.pmean(acc, axis_name="data")
        return loss, acc

    val_step_pmap = eqx.filter_pmap(
        val_step,
        axis_name="data",
        devices=device_list,
    )  # type: ignore[call-overload]

    params_repl = _replicate_tree(params, num_devices)
    params_repl = jax.device_put(params_repl, device=data_sharding)
    bundle_repl = eqx.combine(params_repl, static)
    opt_state_repl = _replicate_tree(opt_state, num_devices)
    opt_state_repl = jax.device_put(opt_state_repl, device=data_sharding)

    global_batch = args.per_device_batch_size * num_devices
    train_steps = len(train_dataset) // global_batch
    val_steps = len(val_dataset) // global_batch
    if train_steps <= 0:
        raise ValueError("train dataset too small for the given batch size")
    if val_steps <= 0:
        raise ValueError("val dataset too small for the given batch size")
    if args.max_train_steps > 0:
        train_steps = min(train_steps, args.max_train_steps)
    if args.max_val_steps > 0:
        val_steps = min(val_steps, args.max_val_steps)

    global_step = 0
    val_global_step = 0
    perf_steps = 0
    perf_data_time = 0.0
    perf_compute_time = 0.0
    perf_log_time = 0.0
    perf_warmup = args.profile_warmup_steps
    last_loss_val = 0.0
    last_acc_val = 0.0

    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng(args.seed + epoch)
        train_iter_host = _iter_epoch_batches(
            train_dataset,
            batch_size=global_batch,
            rng=rng,
            shuffle=args.shuffle,
        )
        train_iter_host = _iter_batches(
            train_iter_host,
            batch_size=global_batch,
            max_seq_len=max_seq_len,
            num_devices=num_devices,
            per_device_batch=args.per_device_batch_size,
        )
        train_iter = _prefetch_to_device(train_iter_host, size=args.prefetch, sharding=data_sharding)
        train_iter = iter(train_iter)

        with tqdm(total=train_steps, desc=f"Train {epoch}/{args.epochs}") as pbar:
            for _ in range(train_steps):
                step_start = time.perf_counter()
                batch_tokens, batch_mask, batch_labels = next(train_iter)
                data_done = time.perf_counter()
                step_key = jax.random.fold_in(base_key, global_step)
                device_keys = jax.random.split(step_key, num_devices)
                device_keys = jax.device_put(device_keys, device=data_sharding)
                if args.train_backbone_after is None:
                    train_backbone = False
                elif args.train_backbone_after <= 0:
                    train_backbone = True
                else:
                    train_backbone = global_step >= args.train_backbone_after

                bundle_repl, opt_state_repl, loss, acc = train_step_pmap(
                    bundle_repl,
                    opt_state_repl,
                    batch_tokens,
                    batch_mask,
                    batch_labels,
                    device_keys,
                    train_backbone,
                )
                if args.profile:
                    jax.block_until_ready(loss)
                compute_done = time.perf_counter()

                log_this_step = (global_step % args.log_every) == 0
                if log_this_step:
                    loss_val = float(np.mean(jax.device_get(loss)))
                    acc_val = float(np.mean(jax.device_get(acc)))
                    last_loss_val = loss_val
                    last_acc_val = acc_val
                    writer.add_scalar("train/loss", loss_val, global_step)
                    writer.add_scalar("train/accuracy", acc_val, global_step)

                if log_this_step:
                    pbar.set_postfix(
                        loss=f"{last_loss_val:.4f}",
                        acc=f"{last_acc_val:.4f}",
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

        val_rng = np.random.default_rng(args.seed + epoch)
        val_iter_host = _iter_epoch_batches(
            val_dataset,
            batch_size=global_batch,
            rng=val_rng,
            shuffle=False,
        )
        val_iter_host = _iter_batches(
            val_iter_host,
            batch_size=global_batch,
            max_seq_len=max_seq_len,
            num_devices=num_devices,
            per_device_batch=args.per_device_batch_size,
        )
        val_iter = _prefetch_to_device(val_iter_host, size=args.prefetch, sharding=data_sharding)
        val_iter = iter(val_iter)

        with tqdm(total=val_steps, desc=f"Val {epoch}/{args.epochs}") as pbar:
            for _ in range(val_steps):
                batch_tokens, batch_mask, batch_labels = next(val_iter)
                loss, acc = val_step_pmap(
                    bundle_repl,
                    batch_tokens,
                    batch_mask,
                    batch_labels,
                )
                loss_val = float(np.mean(jax.device_get(loss)))
                acc_val = float(np.mean(jax.device_get(acc)))
                writer.add_scalar("val/loss", loss_val, val_global_step)
                writer.add_scalar("val/accuracy", acc_val, val_global_step)
                pbar.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc_val:.4f}")
                pbar.update(1)
                val_global_step += 1

        bundle_host = cast(SentimentBundle, _unreplicate(bundle_repl))
        bundle_host = cast(SentimentBundle, _to_host(bundle_host))
        opt_state_host = cast(MuonWithAdamWFallbackState, _unreplicate(opt_state_repl))
        metadata = {
            "epoch": epoch,
            "global_step": global_step,
            "val_global_step": val_global_step,
            "config": run_config,
        }
        _save_checkpoint(run_dir, epoch, bundle_host, opt_state_host, metadata)
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

    writer.close()


if __name__ == "__main__":
    main()
