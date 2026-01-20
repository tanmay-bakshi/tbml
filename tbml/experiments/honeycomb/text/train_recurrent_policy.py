import argparse
import json
import math
import os
import queue
import threading
import time
import traceback
from datetime import datetime
from typing import ClassVar, Iterable, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding
from pydantic import BaseModel, Field
from tensorboardX import SummaryWriter  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from tbml.experiments.honeycomb.text.dataset import MMapTokenDataset, iter_text_batches
from tbml.experiments.honeycomb.text.model import TextTransformer, TextTransformerConfig
from tbml.nn import LSTMStack, Linear
from tbml.nn.init import truncated_normal_init
from tbml.optimizers import MuonWithAdamWFallback, MuonWithAdamWFallbackState, build_muon_masks


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a recurrent policy head on frozen token embeddings."
    )
    parser.add_argument("--runs-folder", type=str, required=True)
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
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

    parser.add_argument("--policy-hidden-dim", type=int, default=None)
    parser.add_argument("--policy-n-layers", type=int, default=None)
    parser.add_argument("--policy-init-std", type=float, default=None)

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
    """Flatten parameter names aligned to leaves.

    :param params: Parameter PyTree.
    :returns: Flat list of parameter names.
    """
    names: list[str] = []
    for path, _ in jax.tree_util.tree_leaves_with_path(params):
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

    jax.tree_util.tree_map(_block, tree)


def _add_trees(tree_a: T, tree_b: T) -> T:
    """Add two PyTrees together.

    :param tree_a: First PyTree.
    :param tree_b: Second PyTree.
    :returns: Sum of the PyTrees.
    """

    def _add(a: object, b: object) -> object:
        if a is None:
            return None
        if b is None:
            return None
        return cast(Array, a) + cast(Array, b)

    return cast(T, jax.tree_util.tree_map(_add, tree_a, tree_b))


def _scale_tree(tree: T, scale: int) -> T:
    """Scale a PyTree by a scalar value.

    :param tree: PyTree of arrays.
    :param scale: Scalar scale factor.
    :returns: Scaled PyTree.
    """
    scale_value = jnp.asarray(scale, dtype=jnp.float32)
    return cast(
        T,
        jax.tree_util.tree_map(
            lambda value: value / scale_value if value is not None else None,
            tree,
        ),
    )


def _cast_tree_dtype(tree: T, dtype: jnp.dtype) -> T:
    """Cast a PyTree of arrays to the requested dtype.

    :param tree: PyTree of arrays.
    :param dtype: Target dtype.
    :returns: PyTree with casted arrays.
    """
    return cast(
        T,
        jax.tree_util.tree_map(
            lambda value: value.astype(dtype) if value is not None else None,
            tree,
        ),
    )


def _build_sharding(devices: list[jax.Device]) -> tuple[Mesh, NamedSharding, NamedSharding]:
    """Build a data-parallel sharding setup.

    :param devices: Device list.
    :returns: Tuple of (mesh, data sharding, replicated sharding).
    """
    mesh = Mesh(np.asarray(devices), ("data",))
    data_sharding = NamedSharding(mesh, PartitionSpec("data"))
    replicated = NamedSharding(mesh, PartitionSpec())
    return mesh, data_sharding, replicated


def _devices_for_platform(platform: str) -> list[jax.Device]:
    """Return devices for a given JAX platform.

    :param platform: Platform name ("cpu" or "gpu").
    :returns: List of devices.
    """
    return [device for device in jax.devices() if device.platform == platform]


def _prefetch_to_device(
    iterator: Iterable[np.ndarray],
    *,
    size: int,
    sharding: Sharding,
) -> Iterable[Array]:
    """Prefetch batches to devices using a background thread.

    :param iterator: Host-side batch iterator.
    :param size: Prefetch buffer size.
    :param sharding: Target device sharding.
    :returns: Iterable of device arrays.
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


def _load_checkpoint_model(
    checkpoint_dir: str,
    *,
    dtype: jnp.dtype,
    model_config: dict[str, object],
) -> TextTransformer:
    """Load a TextTransformer model from a checkpoint.

    :param checkpoint_dir: Checkpoint directory path.
    :param dtype: Compute dtype to try first.
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
    raise RuntimeError("failed to load model")


class RecurrentPolicyConfig(BaseModel):
    """Configuration for the recurrent policy network.

    :ivar vocab_size: Output vocabulary size.
    :ivar input_dim: Input feature dimension.
    :ivar hidden_dim: LSTM hidden state dimension.
    :ivar n_layers: Number of LSTM layers.
    :ivar init_std: Standard deviation for truncated normal initialization.
    """

    vocab_size: int = Field(default=50257)
    input_dim: int = Field(default=768)
    hidden_dim: int = Field(default=768)
    n_layers: int = Field(default=2)
    init_std: float = Field(default=0.02)


class RecurrentPolicy(eqx.Module):
    """Recurrent policy network that predicts tokens from LSTM state."""

    MUON_PARAM_EXCLUSION_PATTERNS: ClassVar[list[str]] = [
        r"^.*norm\..*$",
    ]

    config: RecurrentPolicyConfig = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    rnn: LSTMStack
    head: Linear

    def __init__(
        self,
        config: RecurrentPolicyConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        key: Array,
    ) -> None:
        """Initialize the recurrent policy network.

        :param config: Recurrent policy configuration.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param key: PRNG key for parameter initialization.
        """
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if config.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if config.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if config.init_std <= 0.0:
            raise ValueError("init_std must be > 0")

        init = truncated_normal_init(config.init_std)
        rnn_key, head_key = jax.random.split(key, 2)

        self.config = config
        self.dtype = dtype
        self.rnn = LSTMStack(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.n_layers,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            key=rnn_key,
        )
        self.head = Linear(
            in_features=config.hidden_dim,
            out_features=config.vocab_size,
            use_bias=True,
            bias_value=0.0,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            key=head_key,
        )

    def __call__(self, reps: Array, *, train: bool, key: Array | None) -> Array:
        """Compute logits from token representations.

        :param reps: Token representations of shape (B, T, input_dim).
        :param train: Whether to enable dropout (unused).
        :param key: PRNG key (unused).
        :returns: Logits of shape (B, T, vocab_size).
        """
        if reps.ndim != 3:
            raise ValueError("reps must have shape (B, T, input_dim)")
        if reps.shape[2] != self.config.input_dim:
            raise ValueError("reps last dimension must match input_dim")
        _ = train
        _ = key
        hidden = self.rnn(reps)
        return self.head(hidden)


def _save_checkpoint(
    run_dir: str,
    step: int,
    model: RecurrentPolicy,
    opt_state: MuonWithAdamWFallbackState,
    metadata: dict[str, object],
) -> None:
    """Save model, optimizer, and metadata to disk.

    :param run_dir: Run directory path.
    :param step: Global step index.
    :param model: Policy model to save.
    :param opt_state: Optimizer state to save.
    :param metadata: Metadata to serialize with the checkpoint.
    """
    ckpt_dir = os.path.join(run_dir, f"checkpoint_{step:08d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "model.eqx"), model)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "opt_state.eqx"), opt_state)
    with open(os.path.join(ckpt_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _resolve_optional(value: int | float | None, fallback: int | float) -> int | float:
    """Resolve optional override values.

    :param value: Optional value.
    :param fallback: Fallback when value is None.
    :returns: Resolved value.
    """
    if value is None:
        return fallback
    return value


def main() -> None:
    """Run the recurrent policy training loop."""
    args = _parse_args()
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
    if args.per_device_batch_size <= 0:
        raise ValueError("per-device-batch-size must be > 0")
    if args.num_devices < 0:
        raise ValueError("num-devices must be >= 0")
    if args.prefetch <= 0:
        raise ValueError("prefetch must be > 0")
    if args.shuffle_buffer < 0:
        raise ValueError("shuffle-buffer must be >= 0")

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
    vocab_size = int(metadata.get("vocab_size", 0))
    max_seq_len = int(metadata.get("max_seq_len", 0))
    pad_id = int(metadata.get("pad_id", -1))
    eos_id = int(metadata.get("eos_id", -1))
    if vocab_size <= 0:
        raise ValueError("metadata missing vocab_size")
    if max_seq_len <= 0:
        raise ValueError("metadata missing max_seq_len")
    if pad_id < 0:
        raise ValueError("metadata missing pad_id")
    if eos_id < 0:
        raise ValueError("metadata missing eos_id")
    total_samples = len(dataset)

    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint)
    base_config = _load_run_config(checkpoint_dir)
    model_config_raw = base_config.get("model")
    if model_config_raw is None or isinstance(model_config_raw, dict) is False:
        raise ValueError("checkpoint config missing model entry")
    training_config = base_config.get("training")
    if training_config is None or isinstance(training_config, dict) is False:
        raise ValueError("checkpoint config missing training entry")

    base_model_dtype_name = training_config.get("dtype")
    if isinstance(base_model_dtype_name, str) is False:
        raise ValueError("checkpoint training config missing dtype")
    base_model_dtype = _dtype_from_name(base_model_dtype_name)
    base_model = _load_checkpoint_model(
        checkpoint_dir,
        dtype=base_model_dtype,
        model_config=model_config_raw,
    )
    if base_model.config.vocab_size != vocab_size:
        raise ValueError("base model vocab_size must match dataset")
    if base_model.config.max_seq_len != max_seq_len:
        raise ValueError("base model max_seq_len must match dataset")

    policy_config = RecurrentPolicyConfig(
        vocab_size=vocab_size,
        input_dim=base_model.config.d_model,
        hidden_dim=int(_resolve_optional(args.policy_hidden_dim, base_model.config.d_model)),
        n_layers=int(_resolve_optional(args.policy_n_layers, base_model.config.n_layers)),
        init_std=float(_resolve_optional(args.policy_init_std, base_model.config.init_std)),
    )

    run_dir = _build_run_dir(args.runs_folder)
    exclusion_patterns = list(RecurrentPolicy.MUON_PARAM_EXCLUSION_PATTERNS)

    run_config: dict[str, object] = {
        "policy_type": "recurrent_lstm",
        "policy_model": policy_config.model_dump(),
        "base_checkpoint": checkpoint_dir,
        "data": {
            "dataset_folder": args.data_folder,
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "pad_id": pad_id,
            "eos_id": eos_id,
            "shuffle_buffer": args.shuffle_buffer,
            "num_samples": total_samples,
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
    policy_model = RecurrentPolicy(
        policy_config,
        dtype=dtype,
        param_dtype=param_dtype,
        key=model_key,
    )
    params, static = eqx.partition(policy_model, eqx.is_array)
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
    opt_state = optimizer.init(params)

    def grad_step(
        model_in: RecurrentPolicy,
        base_in: TextTransformer,
        batch_tokens: Array,
        key: Array,
        micro_step0: Array,
        micro_steps: Array,
    ) -> tuple[eqx.Module, Array, Array]:
        """Compute accumulated gradients and metrics across micro-steps.

        :param model_in: Replicated policy model.
        :param base_in: Replicated frozen base model.
        :param batch_tokens: Token batches of shape (M, B, T).
        :param key: PRNG key for dropout.
        :param micro_step0: Starting micro-step index for RNG folding.
        :param micro_steps: Number of valid micro-steps in the batch.
        :returns: Tuple of (grads, loss, accuracy).
        """

        def _loss_fn(
            policy_inner: RecurrentPolicy,
            base_inner: TextTransformer,
            tokens: Array,
            tokens_key: Array,
        ) -> tuple[Array, Array]:
            """Compute the token prediction loss.

            :param policy_inner: Policy model replica used for the loss computation.
            :param base_inner: Frozen base model replica.
            :param tokens: Token batch of shape (B, T).
            :param tokens_key: PRNG key for dropout.
            :returns: Tuple of (loss, accuracy).
            """
            _ = tokens_key
            clean_tokens = jnp.where(tokens == eos_id, pad_id, tokens)
            attention_mask = clean_tokens != pad_id
            reps, _pooled = base_inner(clean_tokens, attention_mask, train=False, key=None)
            reps = jax.lax.stop_gradient(reps)
            reps = reps.astype(policy_inner.dtype)
            logits = policy_inner(reps, train=True, key=tokens_key)
            logits = logits.astype(jnp.float32)

            targets = tokens.reshape((-1,)).astype(jnp.int32)
            mask = jnp.logical_and(targets != pad_id, targets != eos_id)
            logits_flat = logits.reshape((-1, logits.shape[-1]))
            log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
            nll = -jnp.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(axis=-1)
            nll = nll * mask
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            loss = jnp.sum(nll) / denom

            preds = jnp.argmax(logits_flat, axis=-1)
            acc = jnp.sum((preds == targets) & mask) / denom
            return loss, acc

        value_and_grad = eqx.filter_value_and_grad(_loss_fn, has_aux=True)
        params_only = eqx.filter(model_in, eqx.is_array)
        grad_init = jax.tree_util.tree_map(
            lambda value: jnp.zeros_like(value) if value is not None else None,
            params_only,
        )
        loss_init = jnp.asarray(0.0, dtype=jnp.float32)
        acc_init = jnp.asarray(0.0, dtype=jnp.float32)

        step_indices = jnp.arange(batch_tokens.shape[0], dtype=jnp.int32)

        def _accum_body(
            carry: tuple[eqx.Module, Array, Array],
            inputs: tuple[Array, Array],
        ) -> tuple[tuple[eqx.Module, Array, Array], None]:
            """Accumulate gradients and metrics for one micro-step.

            :param carry: Tuple of (grads, loss, accuracy).
            :param inputs: Tuple of (micro step index, token batch).
            :returns: Updated carry and unused output.
            """
            grads_acc, loss_acc, acc_acc = carry
            step_idx, tokens = inputs
            step_idx_global = micro_step0 + step_idx
            tokens_key = jax.random.fold_in(key, step_idx_global)

            def _do_compute(_: None) -> tuple[eqx.Module, Array, Array]:
                (loss, acc), grads = value_and_grad(
                    model_in,
                    base_in,
                    tokens,
                    tokens_key,
                )
                grads = _cast_tree_dtype(grads, jnp.float32)
                grads_accum = _add_trees(grads_acc, grads)
                loss_accum = loss_acc + loss
                acc_accum = acc_acc + acc
                return grads_accum, loss_accum, acc_accum

            def _skip_compute(_: None) -> tuple[eqx.Module, Array, Array]:
                return grads_acc, loss_acc, acc_acc

            active = step_idx < micro_steps
            new_carry = jax.lax.cond(active, _do_compute, _skip_compute, operand=None)
            return new_carry, None

        (grads, loss, acc), _ = jax.lax.scan(
            _accum_body,
            (grad_init, loss_init, acc_init),
            (step_indices, batch_tokens),
        )
        scale = jnp.asarray(micro_steps, dtype=jnp.float32)
        scale = jnp.maximum(scale, 1.0)
        grads = jax.tree_util.tree_map(
            lambda value: value / scale if value is not None else None,
            grads,
        )
        loss = loss / scale
        acc = acc / scale

        grads = jax.lax.pmean(grads, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")
        acc = jax.lax.pmean(acc, axis_name="data")
        return grads, loss, acc

    def apply_step(
        model_in: RecurrentPolicy,
        state_in: MuonWithAdamWFallbackState,
        grads: eqx.Module,
    ) -> tuple[RecurrentPolicy, MuonWithAdamWFallbackState]:
        """Apply gradients to the policy model.

        :param model_in: Replicated policy model.
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
    train_repl = eqx.combine(params_repl, static)
    opt_state_repl = _replicate_tree(opt_state, num_devices)
    opt_state_repl = jax.device_put(opt_state_repl, device=data_sharding)

    base_params, base_static = eqx.partition(base_model, eqx.is_array)
    base_params_repl = _replicate_tree(base_params, num_devices)
    base_params_repl = jax.device_put(base_params_repl, device=data_sharding)
    base_repl = eqx.combine(base_params_repl, base_static)

    global_batch = args.per_device_batch_size * num_devices
    global_step = 0
    perf_steps = 0
    perf_data_time = 0.0
    perf_compute_time = 0.0
    perf_log_time = 0.0
    perf_warmup = args.profile_warmup_steps
    last_loss_val = 0.0
    last_acc_val = 0.0
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

                micro_step0 = jnp.asarray(micro_step, dtype=jnp.int32)
                micro_steps_arr = jnp.asarray(micro_steps, dtype=jnp.int32)
                step_key, base_key = jax.random.split(base_key)

                data_start = time.perf_counter()
                micro_batches: list[Array] = []
                for _ in range(micro_steps):
                    micro_batches.append(next(train_iter))
                data_time = time.perf_counter() - data_start

                batch_tokens = jnp.stack(micro_batches, axis=0)
                batch_tokens = jnp.transpose(batch_tokens, (1, 0, 2, 3))
                batch_tokens = jax.device_put(batch_tokens, device=data_sharding)
                device_keys = jax.random.split(step_key, num_devices)
                device_keys = jax.device_put(device_keys, device=data_sharding)

                compute_start = time.perf_counter()
                grads, loss, acc = grad_step_pmap(
                    train_repl,
                    base_repl,
                    batch_tokens,
                    device_keys,
                    micro_step0,
                    micro_steps_arr,
                )
                compute_time = time.perf_counter() - compute_start
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
                        perf_data_time += data_time
                        perf_compute_time += compute_time
                        perf_log_time += log_done - log_start

                if global_step % args.checkpoint_every == 0:
                    jax.block_until_ready(loss)
                    _block_until_ready(train_repl)
                    _block_until_ready(opt_state_repl)
                    opt_state_host = cast(MuonWithAdamWFallbackState, _unreplicate(opt_state_repl))
                    opt_state_host = cast(MuonWithAdamWFallbackState, _to_host(opt_state_host))
                    model_host = cast(RecurrentPolicy, _unreplicate(train_repl))
                    model_host = cast(RecurrentPolicy, _to_host(model_host))
                    metadata = {"global_step": global_step, "config": run_config}
                    _save_checkpoint(run_dir, global_step, model_host, opt_state_host, metadata)

                if args.profile and perf_steps > 0 and global_step % args.log_every == 0:
                    data_avg = perf_data_time / perf_steps
                    compute_avg = perf_compute_time / perf_steps
                    log_avg = perf_log_time / perf_steps
                    writer.add_scalar("perf/data_seconds", data_avg, global_step)
                    writer.add_scalar("perf/compute_seconds", compute_avg, global_step)
                    writer.add_scalar("perf/log_seconds", log_avg, global_step)
    finally:
        dataset.close()
        writer.close()


if __name__ == "__main__":
    main()
