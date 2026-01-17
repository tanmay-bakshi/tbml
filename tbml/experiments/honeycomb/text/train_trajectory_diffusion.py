import argparse
import json
import math
import os
import queue
import threading
import time
import traceback
from datetime import datetime
from typing import ClassVar, Iterable, Iterator, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding
from pydantic import BaseModel, Field
from tensorboardX import SummaryWriter  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from tbml.experiments.honeycomb.text.dataset import MMapTokenDataset
from tbml.experiments.honeycomb.text.model import TextTransformer, TextTransformerConfig
from tbml.nn import AdaptiveRMSNorm, DropPath, Linear, PoPESelfAttention, RoPESelfAttention, SwiGLUFeedForward
from tbml.nn.init import Initializer, truncated_normal_init
from tbml.optimizers import MuonWithAdamWFallback, MuonWithAdamWFallbackState, build_muon_masks


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train a diffusion model over token trajectories.")
    parser.add_argument("--runs-folder", type=str, required=True)
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--num-prefix-tokens", type=int, required=True)
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

    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--resid-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--attn-type", type=str, default="pope", choices=["pope", "rope"])
    parser.add_argument("--pope-base", type=float, default=10000.0)
    parser.add_argument("--init-std", type=float, default=0.02)

    parser.add_argument("--num-diffusion-steps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--cond-drop-prob", type=float, default=0.1)

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


def _length_from_pad(sample: np.ndarray, pad_id: int, max_seq_len: int) -> int:
    """Compute sequence length from padding.

    :param sample: Token array of shape (T,).
    :param pad_id: Padding token id.
    :param max_seq_len: Maximum sequence length.
    :returns: Non-padding length.
    """
    pad_positions = np.where(sample == pad_id)[0]
    if pad_positions.shape[0] > 0:
        return int(pad_positions[0])
    return max_seq_len


def _iter_filtered_batches(
    dataset: MMapTokenDataset,
    *,
    batch_size: int,
    max_seq_len: int,
    shuffle_buffer: int,
    seed: int,
    num_devices: int,
    per_device_batch: int,
    min_tokens: int,
    pad_id: int,
) -> Iterable[np.ndarray]:
    """Yield filtered and sharded batches with minimum token length.

    :param dataset: Dataset instance.
    :param batch_size: Global batch size.
    :param max_seq_len: Maximum sequence length.
    :param shuffle_buffer: Shuffle buffer size.
    :param seed: Random seed for shuffle order.
    :param num_devices: Number of devices.
    :param per_device_batch: Batch size per device.
    :param min_tokens: Minimum token length to keep a sample.
    :param pad_id: Padding token id.
    :returns: Iterable of sharded token batches.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if shuffle_buffer < 0:
        raise ValueError("shuffle_buffer must be >= 0")
    if min_tokens <= 0:
        raise ValueError("min_tokens must be > 0")

    total = len(dataset)
    rng = np.random.default_rng(seed)
    buffer: list[int] = []

    def _yield_indices() -> Iterable[int]:
        for idx in range(total):
            if shuffle_buffer == 0:
                yield idx
            else:
                if len(buffer) < shuffle_buffer:
                    buffer.append(idx)
                    continue
                pick = int(rng.integers(0, shuffle_buffer))
                yield buffer[pick]
                buffer[pick] = idx
        if shuffle_buffer > 0 and len(buffer) > 0:
            rng.shuffle(buffer)
            for idx in buffer:
                yield idx

    batch: list[np.ndarray] = []
    for idx in _yield_indices():
        sample = dataset[idx]
        length = _length_from_pad(sample, pad_id, max_seq_len)
        if length < min_tokens:
            continue
        batch.append(sample)
        if len(batch) >= batch_size:
            stacked = np.stack(batch, axis=0).astype(np.int32, copy=False)
            stacked = stacked.reshape((num_devices, per_device_batch, max_seq_len))
            yield stacked
            batch = []


def _prepare_prefix_batch_jax(
    tokens: Array,
    *,
    num_prefix: int,
    eos_id: int,
    pad_id: int,
) -> tuple[Array, Array]:
    """Prepare truncated prefix tokens for the base model (JAX version).

    :param tokens: Token ids of shape (B, T).
    :param num_prefix: Number of prefix tokens to keep.
    :param eos_id: EOS token id.
    :param pad_id: Padding token id.
    :returns: Tuple of (prefix_tokens, attention_mask).
    """
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (B, T)")
    if num_prefix <= 0:
        raise ValueError("num_prefix must be > 0")
    batch_size, seq_len = tokens.shape
    prefix_tokens = jnp.full((batch_size, seq_len), pad_id, dtype=tokens.dtype)
    prefix_tokens = prefix_tokens.at[:, :num_prefix].set(tokens[:, :num_prefix])
    attention_mask = jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
    if num_prefix < seq_len:
        prefix_tokens = prefix_tokens.at[:, num_prefix].set(eos_id)
        attention_mask = attention_mask.at[:, : num_prefix + 1].set(True)
    else:
        attention_mask = attention_mask.at[:, :num_prefix].set(True)
    return prefix_tokens, attention_mask


def _timestep_embedding(timesteps: Array, dim: int) -> Array:
    """Compute sinusoidal timestep embeddings.

    :param timesteps: Tensor of shape (B,).
    :param dim: Embedding dimension.
    :returns: Embeddings of shape (B, dim).
    """
    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape (B,)")
    if dim <= 0:
        raise ValueError("dim must be > 0")
    half_dim = dim // 2
    freqs = jnp.exp(-math.log(10000.0) * jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim)
    args = timesteps.astype(jnp.float32)[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


class TimestepEmbedder(eqx.Module):
    """Timestep embedding MLP."""

    time_embed_dim: int
    proj1: Linear
    proj2: Linear

    def __init__(
        self,
        time_embed_dim: int,
        out_dim: int,
        *,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        kernel_init: Initializer,
        key: Array,
    ) -> None:
        """Initialize the timestep embedder.

        :param time_embed_dim: Dimension of the sinusoidal embedding.
        :param out_dim: Output embedding dimension.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param kernel_init: Weight initializer.
        :param key: PRNG key for initialization.
        """
        if time_embed_dim <= 0:
            raise ValueError("time_embed_dim must be > 0")
        if out_dim <= 0:
            raise ValueError("out_dim must be > 0")
        key1, key2 = jax.random.split(key, 2)
        self.time_embed_dim = time_embed_dim
        self.proj1 = Linear(
            in_features=time_embed_dim,
            out_features=out_dim,
            use_bias=True,
            bias_value=0.0,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            key=key1,
        )
        self.proj2 = Linear(
            in_features=out_dim,
            out_features=out_dim,
            use_bias=True,
            bias_value=0.0,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            key=key2,
        )

    def __call__(self, timesteps: Array) -> Array:
        """Compute timestep embeddings.

        :param timesteps: Timestep tensor of shape (B,).
        :returns: Embeddings of shape (B, out_dim).
        """
        emb = _timestep_embedding(timesteps, self.time_embed_dim)
        hidden = self.proj1(emb)
        hidden = jax.nn.silu(hidden)
        return self.proj2(hidden)


class DiffusionBlock(eqx.Module):
    """Diffusion transformer block with adaptive conditioning."""

    norm1: AdaptiveRMSNorm
    attn: PoPESelfAttention | RoPESelfAttention
    drop_path1: DropPath
    norm2: AdaptiveRMSNorm
    mlp: SwiGLUFeedForward
    drop_path2: DropPath

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_hidden_dim: int,
        *,
        attn_dropout: float,
        resid_dropout: float,
        drop_path_prob: float,
        attn_type: str,
        pope_base: float,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        qkv_kernel_init: Initializer,
        o_kernel_init: Initializer,
        mlp_kernel_init: Initializer,
        key: Array,
    ) -> None:
        """Initialize the diffusion transformer block.

        :param d_model: Model width.
        :param n_heads: Number of attention heads.
        :param mlp_hidden_dim: Hidden dimension of the MLP.
        :param attn_dropout: Attention dropout probability.
        :param resid_dropout: Residual dropout probability.
        :param drop_path_prob: DropPath probability for this block.
        :param attn_type: Attention type ("pope" or "rope").
        :param pope_base: Base for PoPE/RoPE frequencies.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param qkv_kernel_init: Initializer for Q/K/V projections.
        :param o_kernel_init: Initializer for output projection.
        :param mlp_kernel_init: Initializer for MLP projections.
        :param key: PRNG key for parameter initialization.
        """
        norm1_key, norm2_key, attn_key, mlp_key = jax.random.split(key, 4)

        self.norm1 = AdaptiveRMSNorm(
            d_model,
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.zeros,
            key=norm1_key,
        )
        if attn_type == "pope":
            self.attn = PoPESelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                is_causal=False,
                base=pope_base,
                dtype=dtype,
                param_dtype=param_dtype,
                qkv_kernel_init=qkv_kernel_init,
                o_kernel_init=o_kernel_init,
                key=attn_key,
            )
        elif attn_type == "rope":
            self.attn = RoPESelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                is_causal=False,
                base=pope_base,
                dtype=dtype,
                param_dtype=param_dtype,
                qkv_kernel_init=qkv_kernel_init,
                o_kernel_init=o_kernel_init,
                key=attn_key,
            )
        else:
            raise ValueError("attn_type must be 'pope' or 'rope'")
        self.drop_path1 = DropPath(drop_path_prob)
        self.norm2 = AdaptiveRMSNorm(
            d_model,
            d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.zeros,
            key=norm2_key,
        )
        self.mlp = SwiGLUFeedForward(
            d_model=d_model,
            hidden_dim=mlp_hidden_dim,
            resid_dropout=resid_dropout,
            dtype=dtype,
            param_dtype=param_dtype,
            gate_up_kernel_init=mlp_kernel_init,
            down_kernel_init=mlp_kernel_init,
            key=mlp_key,
        )
        self.drop_path2 = DropPath(drop_path_prob)

    def __call__(self, x: Array, *, cond: Array, train: bool, key: Array | None) -> Array:
        """Apply the diffusion transformer block.

        :param x: Input tensor of shape (B, T, d_model).
        :param cond: Conditioning tensor of shape (B, d_model).
        :param train: Whether to enable dropout.
        :param key: PRNG key for dropout and DropPath.
        :returns: Output tensor of shape (B, T, d_model).
        """
        if key is None:
            attn_key = None
            mlp_key = None
            drop1_key = None
            drop2_key = None
        else:
            attn_key, mlp_key, drop1_key, drop2_key = jax.random.split(key, 4)

        attn_out = self.attn(self.norm1(x, cond), train=train, key=attn_key)
        attn_out = self.drop_path1(attn_out, train=train, key=drop1_key)
        x = x + attn_out

        mlp_out = self.mlp(self.norm2(x, cond), train=train, key=mlp_key)
        mlp_out = self.drop_path2(mlp_out, train=train, key=drop2_key)
        return x + mlp_out


class DiffusionTransformerConfig(BaseModel):
    """Configuration for the trajectory diffusion model.

    :ivar d_model: Model width.
    :ivar n_heads: Number of attention heads.
    :ivar n_layers: Number of transformer blocks.
    :ivar mlp_ratio: Expansion ratio for the MLP hidden dimension.
    :ivar attn_dropout: Attention dropout probability.
    :ivar resid_dropout: Residual dropout probability.
    :ivar drop_path_rate: Stochastic depth rate at the final block.
    :ivar attn_type: Attention type ("pope" or "rope").
    :ivar pope_base: Base for PoPE/RoPE frequency schedule.
    :ivar init_std: Standard deviation for truncated normal initialization.
    :ivar time_embed_dim: Dimension of the sinusoidal time embedding.
    """

    d_model: int = Field(default=768)
    n_heads: int = Field(default=8)
    n_layers: int = Field(default=8)
    mlp_ratio: float = Field(default=4.0)
    attn_dropout: float = Field(default=0.0)
    resid_dropout: float = Field(default=0.0)
    drop_path_rate: float = Field(default=0.0)
    attn_type: str = Field(default="pope")
    pope_base: float = Field(default=10000.0)
    init_std: float = Field(default=0.02)
    time_embed_dim: int = Field(default=256)


class DiffusionTransformer(eqx.Module):
    """Diffusion transformer for trajectory denoising."""

    MUON_PARAM_EXCLUSION_PATTERNS: ClassVar[list[str]] = [
        r"^.*norm\d*\..*$",
        r"^.*attn\.delta$",
    ]

    config: DiffusionTransformerConfig = eqx.field(static=True)
    time_embedder: TimestepEmbedder
    cond_proj: Linear
    null_cond: Array
    blocks: tuple[DiffusionBlock, ...]
    final_norm: AdaptiveRMSNorm
    output_proj: Linear

    def __init__(
        self,
        config: DiffusionTransformerConfig,
        *,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        key: Array,
    ) -> None:
        """Initialize the diffusion transformer.

        :param config: Diffusion transformer configuration.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param key: PRNG key for parameter initialization.
        """
        if config.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if config.n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if config.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if config.mlp_ratio <= 0.0:
            raise ValueError("mlp_ratio must be > 0")
        if config.attn_dropout < 0.0 or config.attn_dropout >= 1.0:
            raise ValueError("attn_dropout must be in [0, 1)")
        if config.resid_dropout < 0.0 or config.resid_dropout >= 1.0:
            raise ValueError("resid_dropout must be in [0, 1)")
        if config.drop_path_rate < 0.0 or config.drop_path_rate >= 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if config.init_std <= 0.0:
            raise ValueError("init_std must be > 0")
        if config.attn_type not in ("pope", "rope"):
            raise ValueError("attn_type must be 'pope' or 'rope'")
        if config.pope_base <= 1.0:
            raise ValueError("pope_base must be > 1.0")
        if config.time_embed_dim <= 0:
            raise ValueError("time_embed_dim must be > 0")

        init = truncated_normal_init(config.init_std)
        keys = jax.random.split(key, 5 + config.n_layers)
        time_key = keys[0]
        cond_key = keys[1]
        null_key = keys[2]
        norm_key = keys[3]
        block_keys = keys[4 : 4 + config.n_layers]
        out_key = keys[-1]

        self.config = config
        self.time_embedder = TimestepEmbedder(
            config.time_embed_dim,
            config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            key=time_key,
        )
        self.cond_proj = Linear(
            in_features=config.d_model,
            out_features=config.d_model,
            use_bias=True,
            bias_value=0.0,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            key=cond_key,
        )
        self.null_cond = init(null_key, (config.d_model,), param_dtype)

        mlp_hidden_dim = int(config.d_model * config.mlp_ratio)
        if mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be > 0")

        drop_rates = _build_drop_rates(config.drop_path_rate, config.n_layers)
        blocks: list[DiffusionBlock] = []
        for idx in range(config.n_layers):
            blocks.append(
                DiffusionBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    attn_dropout=config.attn_dropout,
                    resid_dropout=config.resid_dropout,
                    drop_path_prob=drop_rates[idx],
                    attn_type=config.attn_type,
                    pope_base=config.pope_base,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    qkv_kernel_init=init,
                    o_kernel_init=init,
                    mlp_kernel_init=init,
                    key=block_keys[idx],
                )
            )
        self.blocks = tuple(blocks)
        self.final_norm = AdaptiveRMSNorm(
            config.d_model,
            config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.zeros,
            key=norm_key,
        )
        self.output_proj = Linear(
            in_features=config.d_model,
            out_features=config.d_model,
            use_bias=True,
            bias_value=0.0,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            key=out_key,
        )

    def __call__(
        self,
        x: Array,
        timesteps: Array,
        cond: Array,
        *,
        train: bool,
        key: Array | None,
    ) -> Array:
        """Predict diffusion noise for a trajectory.

        :param x: Noised trajectory of shape (B, T, d_model).
        :param timesteps: Diffusion timesteps of shape (B,).
        :param cond: Conditioning embeddings of shape (B, d_model).
        :param train: Whether to enable dropout.
        :param key: PRNG key for dropout and DropPath.
        :returns: Predicted noise of shape (B, T, d_model).
        """
        if x.ndim != 3:
            raise ValueError("x must have shape (B, T, d_model)")
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape (B,)")
        if cond.ndim != 2:
            raise ValueError("cond must have shape (B, d_model)")
        if x.shape[0] != timesteps.shape[0] or x.shape[0] != cond.shape[0]:
            raise ValueError("batch sizes must match")

        time_emb = self.time_embedder(timesteps)
        cond_emb = time_emb + self.cond_proj(cond)

        if key is None:
            block_keys: list[Array | None] = [None] * len(self.blocks)
        else:
            block_keys = list(jax.random.split(key, len(self.blocks)))

        for block, block_key in zip(self.blocks, block_keys):
            x = block(x, cond=cond_emb, train=train, key=block_key)

        x = self.final_norm(x, cond_emb)
        return self.output_proj(x)


def _build_drop_rates(drop_path_rate: float, total_layers: int) -> list[float]:
    """Build a linear DropPath schedule.

    :param drop_path_rate: DropPath rate at the final block.
    :param total_layers: Total number of transformer blocks.
    :returns: List of DropPath rates, one per block.
    """
    if total_layers <= 0:
        raise ValueError("total_layers must be > 0")
    if total_layers == 1:
        return [drop_path_rate]
    return [drop_path_rate * (idx / (total_layers - 1)) for idx in range(total_layers)]


def _linear_noise_schedule(
    *,
    beta_start: float,
    beta_end: float,
    num_steps: int,
) -> tuple[Array, Array]:
    """Build a linear beta schedule and cumulative alpha products.

    :param beta_start: Starting beta value.
    :param beta_end: Ending beta value.
    :param num_steps: Number of diffusion steps.
    :returns: Tuple of (betas, alpha_cumprod) arrays.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")
    if beta_start <= 0.0 or beta_end <= 0.0:
        raise ValueError("beta values must be > 0")
    if beta_end <= beta_start:
        raise ValueError("beta_end must be > beta_start")

    betas = jnp.linspace(beta_start, beta_end, num_steps, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_cumprod = jnp.cumprod(alphas)
    return betas, alpha_cumprod


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


def _save_checkpoint(
    run_dir: str,
    step: int,
    model: DiffusionTransformer,
    opt_state: MuonWithAdamWFallbackState,
    metadata: dict[str, object],
) -> None:
    """Save model, optimizer, and metadata to disk.

    :param run_dir: Run directory path.
    :param step: Global step index.
    :param model: Diffusion model to save.
    :param opt_state: Optimizer state to save.
    :param metadata: Metadata to serialize with the checkpoint.
    """
    ckpt_dir = os.path.join(run_dir, f"checkpoint_{step:08d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "model.eqx"), model)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "opt_state.eqx"), opt_state)
    with open(os.path.join(ckpt_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    """Run trajectory diffusion training."""
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
    if args.num_prefix_tokens <= 0:
        raise ValueError("num-prefix-tokens must be > 0")
    if args.num_diffusion_steps <= 0:
        raise ValueError("num-diffusion-steps must be > 0")
    if args.cond_drop_prob < 0.0 or args.cond_drop_prob >= 1.0:
        raise ValueError("cond-drop-prob must be in [0, 1)")

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
    if pad_id < 0 or eos_id < 0:
        raise ValueError("metadata missing pad_id/eos_id")
    if args.num_prefix_tokens >= max_seq_len:
        raise ValueError("num-prefix-tokens must be less than max_seq_len")
    total_samples = len(dataset)

    checkpoint_dir = _resolve_checkpoint_dir(args.base_checkpoint)
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
    if base_model.config.embedding_mode != "causal-token":
        raise ValueError("base checkpoint must use embedding_mode='causal-token'")
    if base_model.config.vocab_size != vocab_size:
        raise ValueError("base model vocab_size must match dataset")
    if base_model.config.max_seq_len != max_seq_len:
        raise ValueError("base model max_seq_len must match dataset")

    model_config = DiffusionTransformerConfig(
        d_model=base_model.config.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        attn_dropout=args.attn_dropout,
        resid_dropout=args.resid_dropout,
        drop_path_rate=args.drop_path_rate,
        attn_type=args.attn_type,
        pope_base=args.pope_base,
        init_std=args.init_std,
        time_embed_dim=base_model.config.d_model * 4,
    )

    run_dir = _build_run_dir(args.runs_folder)
    run_config: dict[str, object] = {
        "base_checkpoint": checkpoint_dir,
        "base_model": base_model.config.model_dump(),
        "diffusion_model": model_config.model_dump(),
        "data": {
            "dataset_folder": args.data_folder,
            "max_seq_len": max_seq_len,
            "vocab_size": vocab_size,
            "pad_id": pad_id,
            "eos_id": eos_id,
            "num_samples": total_samples,
            "shuffle_buffer": args.shuffle_buffer,
        },
        "diffusion": {
            "num_prefix_tokens": args.num_prefix_tokens,
            "num_diffusion_steps": args.num_diffusion_steps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "cond_drop_prob": args.cond_drop_prob,
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
        },
        "training": {
            "dtype": args.dtype,
            "grad_accum_steps": args.grad_accum_steps,
            "max_train_steps": args.max_train_steps,
            "per_device_batch_size": args.per_device_batch_size,
            "num_devices": num_devices,
            "seed": args.seed,
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
    model = DiffusionTransformer(
        model_config,
        dtype=dtype,
        param_dtype=param_dtype,
        key=model_key,
    )

    params, static = eqx.partition(model, eqx.is_array)
    flat_names = _flatten_param_names(params)
    exclusion_patterns = list(DiffusionTransformer.MUON_PARAM_EXCLUSION_PATTERNS)
    muon_mask, qkv_mask = build_muon_masks(params, flat_names, exclusion_patterns)

    flat_params, _ = jax.tree_util.tree_flatten(params)
    flat_muon, _ = jax.tree_util.tree_flatten(muon_mask)
    if len(flat_params) != len(flat_muon):
        raise ValueError("muon_mask must align with params")

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
    betas_schedule, alpha_cumprod = _linear_noise_schedule(
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_steps=args.num_diffusion_steps,
    )
    _ = betas_schedule

    def grad_step(
        model_in: DiffusionTransformer,
        base_in: TextTransformer,
        batch_tokens: Array,
        key: Array,
        micro_step0: Array,
        micro_steps: Array,
    ) -> tuple[eqx.Module, Array]:
        """Compute accumulated gradients and metrics across micro-steps.

        :param model_in: Replicated diffusion model.
        :param base_in: Replicated frozen base model.
        :param batch_tokens: Token batches of shape (M, B, T).
        :param key: PRNG key for dropout.
        :param micro_step0: Starting micro-step index for RNG folding.
        :param micro_steps: Number of valid micro-steps in the batch.
        :returns: Tuple of (grads, loss).
        """

        def _loss_fn(
            model_inner: DiffusionTransformer,
            base_inner: TextTransformer,
            tokens: Array,
            tokens_key: Array,
        ) -> Array:
            """Compute diffusion loss for a batch.

            :param model_inner: Diffusion model replica.
            :param base_inner: Frozen base model replica.
            :param tokens: Token batch of shape (B, T).
            :param tokens_key: PRNG key.
            :returns: Loss scalar.
            """
            prefix_tokens, attention_mask = _prepare_prefix_batch_jax(
                tokens,
                num_prefix=args.num_prefix_tokens,
                eos_id=eos_id,
                pad_id=pad_id,
            )
            reps = base_inner.encode_tokens(prefix_tokens, attention_mask, train=False, key=None)
            if base_inner.config.use_final_norm is True:
                reps = base_inner.final_norm(reps)
            reps = jax.lax.stop_gradient(reps)
            x0 = reps[:, : args.num_prefix_tokens, :]
            cond = x0[:, -1, :]

            bsz = x0.shape[0]
            t_key, noise_key, drop_key = jax.random.split(tokens_key, 3)
            timesteps = jax.random.randint(
                t_key,
                (bsz,),
                minval=0,
                maxval=args.num_diffusion_steps,
            )
            alpha_bar = jnp.take(alpha_cumprod, timesteps)
            alpha_bar = alpha_bar[:, None, None]
            noise = jax.random.normal(noise_key, x0.shape, dtype=x0.dtype)
            x_t = jnp.sqrt(alpha_bar) * x0 + jnp.sqrt(1.0 - alpha_bar) * noise

            drop_mask = jax.random.uniform(drop_key, (bsz,), dtype=jnp.float32) < args.cond_drop_prob
            null_cond = model_inner.null_cond.astype(cond.dtype)
            cond = jnp.where(drop_mask[:, None], null_cond[None, :], cond)

            pred = model_inner(x_t, timesteps, cond, train=True, key=tokens_key)
            pred = pred.astype(jnp.float32)
            noise = noise.astype(jnp.float32)
            loss = jnp.mean(jnp.square(pred - noise))
            return loss

        value_and_grad = eqx.filter_value_and_grad(_loss_fn)
        params_only = eqx.filter(model_in, eqx.is_array)
        grad_init = jax.tree_util.tree_map(
            lambda value: jnp.zeros_like(value) if value is not None else None,
            params_only,
        )
        loss_init = jnp.asarray(0.0, dtype=jnp.float32)

        step_indices = jnp.arange(batch_tokens.shape[0], dtype=jnp.int32)

        def _accum_body(
            carry: tuple[eqx.Module, Array],
            inputs: tuple[Array, Array],
        ) -> tuple[tuple[eqx.Module, Array], None]:
            """Accumulate gradients and metrics for one micro-step.

            :param carry: Tuple of (grads, loss).
            :param inputs: Tuple of (micro step index, token batch).
            :returns: Updated carry and unused output.
            """
            grads_acc, loss_acc = carry
            step_idx, tokens = inputs
            step_idx_global = micro_step0 + step_idx
            tokens_key = jax.random.fold_in(key, step_idx_global)

            def _do_compute(_: None) -> tuple[eqx.Module, Array]:
                loss, grads = value_and_grad(model_in, base_in, tokens, tokens_key)
                grads = _cast_tree_dtype(grads, jnp.float32)
                grads_accum = _add_trees(grads_acc, grads)
                loss_accum = loss_acc + loss
                return grads_accum, loss_accum

            def _skip_compute(_: None) -> tuple[eqx.Module, Array]:
                return grads_acc, loss_acc

            active = step_idx < micro_steps
            new_carry = jax.lax.cond(active, _do_compute, _skip_compute, operand=None)
            return new_carry, None

        (grads, loss), _ = jax.lax.scan(
            _accum_body,
            (grad_init, loss_init),
            (step_indices, batch_tokens),
        )

        scale = jnp.asarray(micro_steps, dtype=jnp.float32)
        scale = jnp.maximum(scale, 1.0)
        grads = jax.tree_util.tree_map(
            lambda value: value / scale if value is not None else None,
            grads,
        )
        loss = loss / scale

        grads = jax.lax.pmean(grads, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")
        return grads, loss

    def apply_step(
        model_in: DiffusionTransformer,
        state_in: MuonWithAdamWFallbackState,
        grads: eqx.Module,
    ) -> tuple[DiffusionTransformer, MuonWithAdamWFallbackState]:
        """Apply gradients to the diffusion model.

        :param model_in: Replicated diffusion model.
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
        train_iter_host = _iter_filtered_batches(
            dataset,
            batch_size=global_batch,
            max_seq_len=max_seq_len,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
            num_devices=num_devices,
            per_device_batch=args.per_device_batch_size,
            min_tokens=args.num_prefix_tokens,
            pad_id=pad_id,
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
                data_time = 0.0
                compute_time = 0.0
                step_key = jax.random.fold_in(base_key, global_step)
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
                grads, loss = grad_step_pmap(
                    train_repl,
                    base_repl,
                    batch_tokens,
                    device_keys,
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
                    last_loss_val = loss_val
                    writer.add_scalar("train/loss", loss_val, global_step)

                if log_this_step:
                    pbar.set_postfix(loss=f"{last_loss_val:.4f}")
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
                    model_host = cast(DiffusionTransformer, _unreplicate(train_repl))
                    model_host = cast(DiffusionTransformer, _to_host(model_host))
                    metadata = {
                        "global_step": global_step,
                        "config": run_config,
                    }
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
