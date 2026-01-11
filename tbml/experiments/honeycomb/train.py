import argparse
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Iterable, Iterator, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding
from tensorboardX import SummaryWriter  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from tbml.data import TarredImagesRandomAccessDataset
from tbml.experiments.honeycomb.dataset import LeJEPADataset
from tbml.experiments.honeycomb.loss import lejepa_loss
from tbml.experiments.honeycomb.model import ConViT, ConViTConfig
from tbml.optimizers import MuonWithAdamWFallback, MuonWithAdamWFallbackState, build_muon_masks


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train a ConViT model with LeJEPA.")
    parser.add_argument("--runs-folder", type=str, required=True)
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup-steps", type=int, default=1)
    parser.add_argument("--attn-impl", type=str, default="auto", choices=["auto", "cudnn", "xla"])
    parser.add_argument("--per-device-batch-size", type=int, default=32)
    parser.add_argument("--num-devices", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device-normalize", dest="device_normalize", action="store_true")
    parser.add_argument(
        "--no-device-normalize",
        dest="device_normalize",
        action="store_false",
    )
    parser.add_argument("--device-augment", dest="device_augment", action="store_true")
    parser.add_argument(
        "--no-device-augment",
        dest="device_augment",
        action="store_false",
    )
    parser.add_argument("--image-size", type=str, default="224,224")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=9)
    parser.add_argument("--n-gpsa-layers", type=int, default=10)
    parser.add_argument("--n-sa-layers", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--resid-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--locality-strength", type=float, default=1.0)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--use-cls-token", dest="use_cls_token", action="store_true")
    parser.add_argument("--no-use-cls-token", dest="use_cls_token", action="store_false")

    parser.add_argument("--resize-dim", type=int, default=256)
    parser.add_argument("--num-global-views", type=int, default=2)
    parser.add_argument("--global-view-dim", type=int, default=224)
    parser.add_argument("--num-local-views", type=int, default=6)
    parser.add_argument("--local-view-dim", type=int, default=96)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--mean", type=str, default="0.485,0.456,0.406")
    parser.add_argument("--std", type=str, default="0.229,0.224,0.225")

    parser.add_argument("--sigreg-weight", type=float, default=0.25)
    parser.add_argument("--sigreg-slices", type=int, default=256)
    parser.add_argument("--sigreg-seed", type=int, default=0)

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
        device_normalize=True,
        device_augment=True,
        use_cls_token=False,
    )
    return parser.parse_args()


def _parse_pair(value: str) -> tuple[int, int]:
    """Parse a comma-separated integer pair.

    :param value: Input string of the form "a,b".
    :returns: Parsed integer pair.
    """
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("expected two comma-separated values")
    return int(parts[0]), int(parts[1])


def _parse_triple(value: str) -> tuple[float, float, float]:
    """Parse a comma-separated float triple.

    :param value: Input string of the form "a,b,c".
    :returns: Parsed float triple.
    """
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError("expected three comma-separated values")
    return float(parts[0]), float(parts[1]), float(parts[2])


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


def _collect_tar_paths(folder: str, prefix: str) -> list[str]:
    """Collect tar file paths with a given prefix.

    :param folder: Directory containing tar files.
    :param prefix: Filename prefix to match.
    :returns: Sorted list of tar file paths.
    """
    if os.path.isdir(folder) is False:
        raise FileNotFoundError(f"data folder not found: {folder}")
    paths: list[str] = []
    for name in sorted(os.listdir(folder)):
        if name.startswith(prefix) is False:
            continue
        if name.endswith(".tar") is False:
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path) is False:
            continue
        paths.append(path)
    if len(paths) == 0:
        raise FileNotFoundError(f"no tar files found for prefix: {prefix}")
    return paths


def _build_run_dir(runs_folder: str) -> str:
    """Create a unique run directory.

    :param runs_folder: Root directory for runs.
    :returns: Newly created run directory path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_folder, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


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


def _build_sharding(devices: list[jax.Device]) -> tuple[Mesh, NamedSharding, NamedSharding]:
    """Build sharding helpers for data and replicated params.

    :param devices: Devices to shard across.
    :returns: Tuple of (mesh, data_sharding, replicated_sharding).
    """
    mesh = Mesh(np.array(devices), ("data",))
    data_sharding = NamedSharding(mesh, PartitionSpec("data"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    return mesh, data_sharding, replicated_sharding


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


def _prefetch_to_device(
    iterator: Iterable[np.ndarray],
    size: int,
    sharding: Sharding,
    *,
    host_cast_dtype: np.dtype | None = None,
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
        try:
            for item in iterator:
                if host_cast_dtype is not None:
                    item = item.astype(host_cast_dtype, copy=False)
                device_item = jax.device_put(item, device=sharding)
                work_queue.put(device_item)
        except Exception as exc:
            work_queue.put(exc)
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
    dataset: LeJEPADataset,
    indices: np.ndarray,
    per_device_batch: int,
    num_devices: int,
    executor: ThreadPoolExecutor | None,
) -> Iterable[np.ndarray]:
    """Yield device-sharded batches from the dataset.

    :param dataset: Dataset to sample from.
    :param indices: Array of indices to iterate over.
    :param per_device_batch: Batch size per device.
    :param num_devices: Number of devices.
    :param executor: Optional executor for parallel loading.
    :returns: Iterable of sharded batches.
    """
    global_batch = per_device_batch * num_devices
    total = (len(indices) // global_batch) * global_batch
    for start in range(0, total, global_batch):
        batch_indices = indices[start : start + global_batch]
        if executor is None:
            items = [dataset[int(idx)] for idx in batch_indices]
        else:
            items = list(executor.map(dataset.__getitem__, batch_indices))
        stacked = np.stack(items, axis=0)
        yield stacked.reshape((num_devices, per_device_batch) + stacked.shape[1:])


def _augment_views(
    images: Array,
    key: Array,
    *,
    num_global_views: int,
    global_view_dim: int,
    num_local_views: int,
    local_view_dim: int,
    resize_locals: bool,
) -> tuple[Array, Array]:
    """Create global/local crops on device.

    :param images: Input images of shape (B, H, W, C).
    :param key: PRNG key for sampling crops.
    :param resize_locals: Whether to resize local crops to the global size.
    :returns: Tuple of (global_views, local_views).
    """
    if images.ndim != 4:
        raise ValueError("images must have shape (B, H, W, C)")
    bsz, height, width, channels = images.shape

    def _random_crops(source: Array, crop_key: Array, num_views: int, crop_dim: int) -> Array:
        if num_views == 0:
            return jnp.zeros((bsz, 0, crop_dim, crop_dim, channels), dtype=source.dtype)
        max_y = height - crop_dim
        max_x = width - crop_dim
        key_y, key_x = jax.random.split(crop_key)
        ys = jax.random.randint(key_y, (bsz, num_views), 0, max_y + 1)
        xs = jax.random.randint(key_x, (bsz, num_views), 0, max_x + 1)

        def _crop_one(img: Array, y: Array, x: Array) -> Array:
            return jax.lax.dynamic_slice(img, (y, x, 0), (crop_dim, crop_dim, channels))

        def _crop_views(img: Array, yv: Array, xv: Array) -> Array:
            return jax.vmap(_crop_one, in_axes=(None, 0, 0))(img, yv, xv)

        return jax.vmap(_crop_views, in_axes=(0, 0, 0))(source, ys, xs)

    key, global_key = jax.random.split(key)
    global_views = _random_crops(images, global_key, num_global_views, global_view_dim)

    if num_local_views > 0:
        key, local_key = jax.random.split(key)
        local_views = _random_crops(images, local_key, num_local_views, local_view_dim)
        if resize_locals:
            local_views = jax.image.resize(
                local_views.astype(jnp.float32),
                (bsz, num_local_views, global_view_dim, global_view_dim, channels),
                method="bilinear",
            )
    else:
        local_views = jnp.zeros(
            (bsz, 0, local_view_dim, local_view_dim, channels),
            dtype=images.dtype,
        )
    return global_views, local_views


def _encode_images(
    model: ConViT,
    views: Array,
    *,
    train: bool,
    key: Array | None,
    dtype: jnp.dtype,
    mean: Array | None,
    std: Array | None,
    normalize_on_device: bool,
) -> Array:
    """Encode same-size views into pooled embeddings.

    :param model: ConViT encoder.
    :param views: Input views of shape (B, V, H, W, C).
    :param train: Whether to enable dropout.
    :param key: PRNG key for dropout.
    :param dtype: Compute dtype.
    :param mean: Optional per-channel mean.
    :param std: Optional per-channel std.
    :param normalize_on_device: Whether to normalize inputs on device.
    :returns: Embeddings of shape (B, V, K).
    """
    if views.ndim != 5:
        raise ValueError("views must have shape (B, V, H, W, C)")
    bsz, num_views, height, width, channels = views.shape
    images = views.reshape((bsz * num_views, height, width, channels))
    if normalize_on_device:
        images = images.astype(jnp.float32)
        if mean is not None and std is not None:
            images = images / jnp.asarray(255.0, dtype=images.dtype)
            images = (images - mean) / std
        images = images.astype(dtype)
    else:
        images = images.astype(dtype)
    pooled = model(images, train=train, key=key)
    return pooled.reshape((bsz, num_views, pooled.shape[-1]))


def _encode_views(
    model: ConViT,
    batch: Array,
    *,
    train: bool,
    key: jax.Array | None,
    dtype: jnp.dtype,
    mean: Array | None,
    std: Array | None,
    normalize_on_device: bool,
    augment_on_device: bool,
    num_global_views: int,
    global_view_dim: int,
    num_local_views: int,
    local_view_dim: int,
) -> Array:
    """Encode a batch of views into pooled embeddings.

    :param model: ConViT encoder.
    :param batch: Input views of shape (B, V, H, W, C) or base images of shape (B, H, W, C).
    :param train: Whether to enable dropout.
    :param key: PRNG key for dropout.
    :param dtype: Compute dtype.
    :param mean: Optional per-channel mean.
    :param std: Optional per-channel std.
    :param normalize_on_device: Whether to normalize inputs on device.
    :param augment_on_device: Whether to generate views on device.
    :param num_global_views: Number of global views to sample.
    :param global_view_dim: Global view crop size.
    :param num_local_views: Number of local views to sample.
    :param local_view_dim: Local view crop size.
    :returns: Embeddings of shape (B, V, K).
    """
    if augment_on_device:
        if key is None:
            raise ValueError("device augmentation requires a PRNG key")
        if batch.ndim != 4:
            raise ValueError("batch must have shape (B, H, W, C) for device augmentation")
        key, aug_key = jax.random.split(key)
        global_views, local_views = _augment_views(
            batch,
            aug_key,
            num_global_views=num_global_views,
            global_view_dim=global_view_dim,
            num_local_views=num_local_views,
            local_view_dim=local_view_dim,
            resize_locals=True,
        )
        if num_local_views > 0 and global_views.shape[2:4] == local_views.shape[2:4]:
            key, encode_key = jax.random.split(key)
            combined = jnp.concatenate([global_views, local_views], axis=1)
            return _encode_images(
                model,
                combined,
                train=train,
                key=encode_key,
                dtype=dtype,
                mean=mean,
                std=std,
                normalize_on_device=normalize_on_device,
            )

        outputs: list[Array] = []
        if num_global_views > 0:
            key, global_key = jax.random.split(key)
            outputs.append(
                _encode_images(
                    model,
                    global_views,
                    train=train,
                    key=global_key,
                    dtype=dtype,
                    mean=mean,
                    std=std,
                    normalize_on_device=normalize_on_device,
                )
            )
        if num_local_views > 0:
            key, local_key = jax.random.split(key)
            outputs.append(
                _encode_images(
                    model,
                    local_views,
                    train=train,
                    key=local_key,
                    dtype=dtype,
                    mean=mean,
                    std=std,
                    normalize_on_device=normalize_on_device,
                )
            )
        if len(outputs) == 1:
            return outputs[0]
        return jnp.concatenate(outputs, axis=1)

    return _encode_images(
        model,
        batch,
        train=train,
        key=key,
        dtype=dtype,
        mean=mean,
        std=std,
        normalize_on_device=normalize_on_device,
    )


def _save_checkpoint(
    run_dir: str,
    epoch: int,
    model: ConViT,
    opt_state: eqx.Module,
    metadata: dict[str, object],
) -> None:
    """Save model, optimizer, and metadata to disk.

    :param run_dir: Run directory path.
    :param epoch: Epoch index.
    :param model: Model to serialize.
    :param opt_state: Optimizer state to serialize.
    :param metadata: Metadata to persist.
    """
    ckpt_dir = os.path.join(run_dir, f"checkpoint_epoch_{epoch:04d}")
    os.makedirs(ckpt_dir, exist_ok=False)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "model.eqx"), model)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "optimizer.eqx"), opt_state)
    with open(os.path.join(ckpt_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    """Run the training loop."""
    args = _parse_args()

    image_size = _parse_pair(args.image_size)
    mean = _parse_triple(args.mean)
    std = _parse_triple(args.std)
    betas = _parse_betas(args.adamw_betas)

    dtype = _dtype_from_name(args.dtype)
    run_dir = _build_run_dir(args.runs_folder)

    device_list: list[jax.Device] = list(jax.devices("gpu"))
    if args.num_devices == 0:
        device_list = list(jax.devices("cpu"))
    if args.num_devices > 0:
        if len(device_list) < args.num_devices:
            raise ValueError("not enough devices available")
        device_list = device_list[: args.num_devices]
    if len(device_list) == 0:
        raise ValueError("no devices available for training")

    num_devices = len(device_list)
    _mesh, data_sharding, _replicated_sharding = _build_sharding(device_list)
    attn_impl = args.attn_impl
    if attn_impl == "auto":
        if any(dev.platform == "gpu" for dev in device_list) and dtype in (
            jnp.float16,
            jnp.bfloat16,
        ):
            attn_impl = "cudnn"
        else:
            attn_impl = "xla"
    if args.use_cls_token is True and attn_impl == "cudnn":
        attn_impl = "xla"
    if args.per_device_batch_size <= 0:
        raise ValueError("per-device batch size must be > 0")
    if args.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if args.max_train_steps < 0:
        raise ValueError("max-train-steps must be >= 0")
    if args.max_val_steps < 0:
        raise ValueError("max-val-steps must be >= 0")
    if args.log_every <= 0:
        raise ValueError("log-every must be > 0")
    if args.profile_warmup_steps < 0:
        raise ValueError("profile-warmup-steps must be >= 0")
    if args.attn_impl not in {"auto", "cudnn", "xla"}:
        raise ValueError("attn-impl must be auto, cudnn, or xla")
    if args.device_augment and args.device_normalize is False:
        raise ValueError("device_augment requires device_normalize to be enabled")
    if args.use_cls_token is True and args.n_sa_layers <= 0:
        raise ValueError("use_cls_token requires at least one SA layer")
    if args.global_view_dim != image_size[0] or args.global_view_dim != image_size[1]:
        raise ValueError("global_view_dim must match image_size for the encoder")
    if args.global_view_dim % args.patch_size != 0:
        raise ValueError("global_view_dim must be divisible by patch_size")
    if args.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if args.prefetch <= 0:
        raise ValueError("prefetch must be > 0")
    if args.sigreg_weight < 0.0 or args.sigreg_weight > 1.0:
        raise ValueError("sigreg_weight must be in [0, 1]")

    effective_prefetch = args.prefetch
    if args.num_workers > 0:
        auto_prefetch = min(args.num_workers * 2, 8)
        effective_prefetch = max(args.prefetch, auto_prefetch)

    train_paths = _collect_tar_paths(args.data_folder, "train")
    val_paths = _collect_tar_paths(args.data_folder, "validation")

    mean_std = None
    if args.no_standardize is False:
        mean_std = (mean, std)
    normalize_on_device = args.device_normalize
    augment_on_device = args.device_augment
    device_mean = None
    device_std = None
    if mean_std is not None and normalize_on_device:
        device_mean = jnp.asarray(mean, dtype=jnp.float32)
        device_std = jnp.asarray(std, dtype=jnp.float32)

    train_base = TarredImagesRandomAccessDataset(train_paths)
    val_base = TarredImagesRandomAccessDataset(val_paths)

    dataset_normalize = (normalize_on_device is False) and (augment_on_device is False)
    train_dataset = LeJEPADataset(
        train_base,
        resize_dim=args.resize_dim,
        num_global_views=args.num_global_views,
        global_view_dim=args.global_view_dim,
        num_local_views=args.num_local_views,
        local_view_dim=args.local_view_dim,
        mean_std=mean_std,
        seed=args.seed,
        normalize=dataset_normalize,
        device_augment=augment_on_device,
    )
    val_dataset = LeJEPADataset(
        val_base,
        resize_dim=args.resize_dim,
        num_global_views=args.num_global_views,
        global_view_dim=args.global_view_dim,
        num_local_views=args.num_local_views,
        local_view_dim=args.local_view_dim,
        mean_std=mean_std,
        seed=args.seed,
        normalize=dataset_normalize,
        device_augment=augment_on_device,
    )

    model_config = ConViTConfig(
        image_size=image_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_gpsa_layers=args.n_gpsa_layers,
        n_sa_layers=args.n_sa_layers,
        mlp_ratio=args.mlp_ratio,
        attn_dropout=args.attn_dropout,
        resid_dropout=args.resid_dropout,
        drop_path_rate=args.drop_path_rate,
        locality_strength=args.locality_strength,
        init_std=args.init_std,
        use_cls_token=args.use_cls_token,
    )

    run_config: dict[str, object] = {
        "model": model_config.model_dump(),
        "attention": {"implementation": attn_impl},
        "data": {
            "resize_dim": args.resize_dim,
            "num_global_views": args.num_global_views,
            "global_view_dim": args.global_view_dim,
            "num_local_views": args.num_local_views,
            "local_view_dim": args.local_view_dim,
            "mean_std": mean_std,
            "device_normalize": args.device_normalize,
            "device_augment": args.device_augment,
        },
        "loss": {
            "sigreg_weight": args.sigreg_weight,
            "sigreg_slices": args.sigreg_slices,
            "sigreg_seed": args.sigreg_seed,
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
            "muon_param_exclusion_patterns": ConViT.MUON_PARAM_EXCLUSION_PATTERNS,
        },
        "training": {
            "epochs": args.epochs,
            "max_train_steps": args.max_train_steps,
            "max_val_steps": args.max_val_steps,
            "per_device_batch_size": args.per_device_batch_size,
            "num_devices": num_devices,
            "seed": args.seed,
            "dtype": args.dtype,
            "log_every": args.log_every,
            "profile": args.profile,
            "profile_warmup_steps": args.profile_warmup_steps,
            "num_workers": args.num_workers,
            "prefetch": effective_prefetch,
        },
    }

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    writer = SummaryWriter(run_dir)
    writer.add_text("config", json.dumps(run_config, indent=2))

    base_key = jax.random.PRNGKey(args.seed)
    model_key, base_key = jax.random.split(base_key)
    model = ConViT(
        model_config,
        dtype=dtype,
        param_dtype=dtype,
        attn_implementation=attn_impl,
        key=model_key,
    )

    model_params, model_static = eqx.partition(model, eqx.is_array)
    flat_names = _flatten_param_names(model_params)
    muon_mask, qkv_mask = build_muon_masks(
        model_params,
        flat_names,
        ConViT.MUON_PARAM_EXCLUSION_PATTERNS,
    )

    flat_params, _ = jax.tree_util.tree_flatten(model_params)
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

    opt_state: MuonWithAdamWFallbackState = optimizer.init(model_params)

    def train_step(
        model_in: ConViT,
        state_in: MuonWithAdamWFallbackState,
        batch: Array,
        key: Array,
        global_step: Array,
    ) -> tuple[ConViT, MuonWithAdamWFallbackState, Array, Array, Array]:
        def _loss_fn(model_inner: ConViT) -> tuple[Array, tuple[Array, Array]]:
            emb = _encode_views(
                model_inner,
                batch,
                train=True,
                key=key,
                dtype=dtype,
                mean=device_mean,
                std=device_std,
                normalize_on_device=normalize_on_device,
                augment_on_device=augment_on_device,
                num_global_views=args.num_global_views,
                global_view_dim=args.global_view_dim,
                num_local_views=args.num_local_views,
                local_view_dim=args.local_view_dim,
            )
            total, pred, sigreg = lejepa_loss(
                emb,
                args.num_global_views,
                sigreg_weight=args.sigreg_weight,
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

    def eval_step(
        model_in: ConViT,
        batch: Array,
        key: Array,
        global_step: Array,
    ) -> tuple[Array, Array, Array]:
        emb = _encode_views(
            model_in,
            batch,
            train=False,
            key=key,
            dtype=dtype,
            mean=device_mean,
            std=device_std,
            normalize_on_device=normalize_on_device,
            augment_on_device=augment_on_device,
            num_global_views=args.num_global_views,
            global_view_dim=args.global_view_dim,
            num_local_views=args.num_local_views,
            local_view_dim=args.local_view_dim,
        )
        total, pred, sigreg = lejepa_loss(
            emb,
            args.num_global_views,
            sigreg_weight=args.sigreg_weight,
            global_step=global_step,
            num_slices=args.sigreg_slices,
            seed=args.sigreg_seed,
            axis_name="data",
        )
        total = jax.lax.pmean(total, axis_name="data")
        pred = jax.lax.pmean(pred, axis_name="data")
        sigreg = jax.lax.pmean(sigreg, axis_name="data")
        return total, pred, sigreg

    train_step_pmap = eqx.filter_pmap(
        train_step,
        axis_name="data",
        devices=device_list,
        donate="all",
    )  # type: ignore[call-overload]
    eval_step_pmap = eqx.filter_pmap(
        eval_step, axis_name="data", devices=device_list
    )  # type: ignore[call-overload]

    model_params_repl = _replicate_tree(model_params, num_devices)
    model_params_repl = jax.device_put(model_params_repl, device=data_sharding)
    model_repl = eqx.combine(model_params_repl, model_static)
    opt_state_repl = _replicate_tree(opt_state, num_devices)
    opt_state_repl = jax.device_put(opt_state_repl, device=data_sharding)

    global_batch = args.per_device_batch_size * num_devices
    train_steps = len(train_dataset) // global_batch
    val_steps = len(val_dataset) // global_batch
    if args.max_train_steps > 0:
        train_steps = min(train_steps, args.max_train_steps)
    if args.max_val_steps > 0:
        val_steps = min(val_steps, args.max_val_steps)
    if train_steps <= 0 or val_steps <= 0:
        raise ValueError("dataset too small for the requested batch size")

    rng = np.random.default_rng(args.seed)
    global_step = 0
    val_global_step = 0
    executor = ThreadPoolExecutor(max_workers=args.num_workers) if args.num_workers > 0 else None
    host_cast_dtype: np.dtype | None = None
    if normalize_on_device is False and dtype != jnp.float32:
        host_cast_dtype = np.float16
    perf_steps = 0
    perf_data_time = 0.0
    perf_compute_time = 0.0
    perf_log_time = 0.0
    perf_warmup = args.profile_warmup_steps
    last_loss_val = 0.0
    last_pred_val = 0.0
    last_sig_val = 0.0

    try:
        for epoch in range(1, args.epochs + 1):
            train_indices = rng.permutation(len(train_dataset))
            if train_steps * global_batch < len(train_indices):
                train_indices = train_indices[: train_steps * global_batch]
            train_iter_host = _iter_batches(
                train_dataset,
                train_indices,
                args.per_device_batch_size,
                num_devices,
                executor,
            )
            train_iter = _prefetch_to_device(
                train_iter_host,
                size=effective_prefetch,
                sharding=data_sharding,
                host_cast_dtype=host_cast_dtype,
            )
            train_iter = iter(train_iter)

            with tqdm(total=train_steps, desc=f"Train {epoch}/{args.epochs}") as pbar:
                for _ in range(train_steps):
                    step_start = time.perf_counter()
                    batch = next(train_iter)
                    data_done = time.perf_counter()
                    step_key = jax.random.fold_in(base_key, global_step)
                    device_keys = jax.random.split(step_key, num_devices)
                    device_keys = jax.device_put(device_keys, device=data_sharding)
                    step_id = jnp.full((num_devices,), global_step, dtype=jnp.int32)

                    model_repl, opt_state_repl, loss, pred, sigreg = train_step_pmap(
                        model_repl,
                        opt_state_repl,
                        batch,
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

                    if log_this_step:
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

            val_indices = np.arange(len(val_dataset))
            if val_steps * global_batch < len(val_indices):
                val_indices = val_indices[: val_steps * global_batch]
            val_iter_host = _iter_batches(
                val_dataset,
                val_indices,
                args.per_device_batch_size,
                num_devices,
                executor,
            )
            val_iter = _prefetch_to_device(
                val_iter_host,
                size=effective_prefetch,
                sharding=data_sharding,
                host_cast_dtype=host_cast_dtype,
            )
            val_iter = iter(val_iter)

            with tqdm(total=val_steps, desc=f"Val {epoch}/{args.epochs}") as pbar:
                for _ in range(val_steps):
                    batch = next(val_iter)
                    step_id = jnp.full((num_devices,), global_step, dtype=jnp.int32)
                    device_keys = jax.random.split(base_key, num_devices)
                    device_keys = jax.device_put(device_keys, device=data_sharding)
                    total, pred, sigreg = eval_step_pmap(model_repl, batch, device_keys, step_id)

                    loss_val = float(np.mean(jax.device_get(total)))
                    pred_val = float(np.mean(jax.device_get(pred)))
                    sig_val = float(np.mean(jax.device_get(sigreg)))

                    writer.add_scalar("val/total_loss", loss_val, val_global_step)
                    writer.add_scalar("val/pred_loss", pred_val, val_global_step)
                    writer.add_scalar("val/sigreg_loss", sig_val, val_global_step)

                    pbar.set_postfix(
                        total=f"{loss_val:.4f}",
                        pred=f"{pred_val:.4f}",
                        sigreg=f"{sig_val:.4f}",
                    )
                    pbar.update(1)
                    val_global_step += 1

            model_host = cast(ConViT, _unreplicate(model_repl))
            opt_state_host = cast(MuonWithAdamWFallbackState, _unreplicate(opt_state_repl))
            metadata = {
                "epoch": epoch,
                "global_step": global_step,
                "val_global_step": val_global_step,
                "config": run_config,
            }
            _save_checkpoint(run_dir, epoch, model_host, opt_state_host, metadata)
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
        if executor is not None:
            executor.shutdown(wait=True)
        writer.close()


if __name__ == "__main__":
    main()
