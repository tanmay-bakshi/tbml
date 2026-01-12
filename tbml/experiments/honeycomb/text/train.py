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
from tbml.experiments.honeycomb.text.dataset import StreamingTextDataset, iter_text_batches
from tbml.experiments.honeycomb.text.model import TextTransformer, TextTransformerConfig
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
    parser.add_argument("--global-mask-min", type=float, default=0.2)
    parser.add_argument("--global-mask-max", type=float, default=0.4)
    parser.add_argument("--local-mask-min", type=float, default=0.6)
    parser.add_argument("--local-mask-max", type=float, default=0.7)

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
        use_cls_token=False,
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
) -> Array:
    """Apply random masking to tokens.

    :param tokens: Token ids of shape (B, T).
    :param key: PRNG key.
    :param min_ratio: Minimum masking ratio.
    :param max_ratio: Maximum masking ratio.
    :param mask_id: Masking token id.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :returns: Masked token ids of shape (B, T).
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
    return masked_tokens


def _mask_views(
    tokens: Array,
    key: Array,
    *,
    num_views: int,
    min_ratio: float,
    max_ratio: float,
    mask_id: int,
    pad_id: int,
    eos_id: int,
) -> Array:
    """Generate multiple masked views of the token batch.

    :param tokens: Token ids of shape (B, T).
    :param key: PRNG key.
    :param num_views: Number of views to generate.
    :param min_ratio: Minimum masking ratio.
    :param max_ratio: Maximum masking ratio.
    :param mask_id: Masking token id.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :returns: Masked views of shape (B, V, T).
    """
    if num_views == 0:
        return jnp.zeros((tokens.shape[0], 0, tokens.shape[1]), dtype=tokens.dtype)

    keys = jax.random.split(key, num_views)

    def _one_view(view_key: Array) -> Array:
        """Mask tokens using the provided view key.

        :param view_key: PRNG key for masking.
        :returns: Masked token ids of shape (B, T).
        """
        return _mask_tokens(
            tokens,
            view_key,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            mask_id=mask_id,
            pad_id=pad_id,
            eos_id=eos_id,
        )

    views = jax.vmap(_one_view)(keys)
    return jnp.transpose(views, (1, 0, 2))


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
    mask_id: int,
    pad_id: int,
    eos_id: int,
) -> Array:
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
    :param mask_id: Masking token id.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :returns: Embeddings of shape (B, V, d_model).
    """
    if num_global_views + num_local_views <= 0:
        raise ValueError("at least one view must be requested")

    key, global_key, local_key = jax.random.split(key, 3)
    global_views = _mask_views(
        tokens,
        global_key,
        num_views=num_global_views,
        min_ratio=global_mask_min,
        max_ratio=global_mask_max,
        mask_id=mask_id,
        pad_id=pad_id,
        eos_id=eos_id,
    )
    local_views = _mask_views(
        tokens,
        local_key,
        num_views=num_local_views,
        min_ratio=local_mask_min,
        max_ratio=local_mask_max,
        mask_id=mask_id,
        pad_id=pad_id,
        eos_id=eos_id,
    )
    if num_local_views > 0 and num_global_views > 0:
        views = jnp.concatenate([global_views, local_views], axis=1)
    elif num_global_views > 0:
        views = global_views
    else:
        views = local_views

    bsz, num_views, seq_len = views.shape
    flat_views = views.reshape((bsz * num_views, seq_len))
    flat_mask = jnp.repeat(attention_mask, repeats=num_views, axis=0)
    pooled = model(flat_views, flat_mask, train=train, key=key)
    return pooled.reshape((bsz, num_views, pooled.shape[-1]))


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

    dtype = _dtype_from_name(args.dtype)
    betas = _parse_betas(args.adamw_betas)
    run_dir = _build_run_dir(args.runs_folder)

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
        num_workers=args.num_workers,
        prefetch=args.prefetch,
        seed=args.seed,
    )
    vocab_size, eos_id, pad_id, mask_id = dataset.tokenizer_info()

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
            "muon_param_exclusion_patterns": TextTransformer.MUON_PARAM_EXCLUSION_PATTERNS,
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

    model_params, model_static = eqx.partition(model, eqx.is_array)
    flat_names = _flatten_param_names(model_params)
    muon_mask, qkv_mask = build_muon_masks(
        model_params,
        flat_names,
        TextTransformer.MUON_PARAM_EXCLUSION_PATTERNS,
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
            emb = _encode_views(
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
                mask_id=mask_id,
                pad_id=pad_id,
                eos_id=eos_id,
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

    train_step_pmap = eqx.filter_pmap(
        train_step,
        axis_name="data",
        devices=device_list,
        donate="all",
    )  # type: ignore[call-overload]

    model_params_repl = _replicate_tree(model_params, num_devices)
    model_params_repl = jax.device_put(model_params_repl, device=data_sharding)
    model_repl = eqx.combine(model_params_repl, model_static)
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

        total_steps = args.max_train_steps if args.max_train_steps > 0 else None
        with tqdm(total=total_steps, desc="Train") as pbar:
            while True:
                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    break
                step_start = time.perf_counter()
                batch_tokens, batch_mask = next(train_iter)
                data_done = time.perf_counter()
                step_key = jax.random.fold_in(base_key, global_step)
                device_keys = jax.random.split(step_key, num_devices)
                device_keys = jax.device_put(device_keys, device=data_sharding)
                step_id = jnp.full((num_devices,), global_step, dtype=jnp.int32)

                model_repl, opt_state_repl, loss, pred, sigreg = train_step_pmap(
                    model_repl,
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

                if global_step % args.checkpoint_every == 0:
                    model_host = cast(TextTransformer, _unreplicate(model_repl))
                    opt_state_host = cast(MuonWithAdamWFallbackState, _unreplicate(opt_state_repl))
                    metadata = {
                        "global_step": global_step,
                        "config": run_config,
                    }
                    _save_checkpoint(run_dir, global_step, model_host, opt_state_host, metadata)
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
