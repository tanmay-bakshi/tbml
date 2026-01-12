import math
from functools import lru_cache
import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from tbml.nn.linear import Linear
from tbml.nn.init import Initializer


def _build_rel_pos(grid_size: tuple[int, int]) -> np.ndarray:
    """Build relative position features for a square patch grid."""
    coords_h = np.arange(grid_size[0], dtype=np.float32)
    coords_w = np.arange(grid_size[1], dtype=np.float32)
    grid_h, grid_w = np.meshgrid(coords_h, coords_w, indexing="ij")
    coords = np.stack((grid_h, grid_w), axis=-1).reshape((grid_size[0] * grid_size[1], 2))
    delta = coords[None, :, :] - coords[:, None, :]
    delta1 = delta[..., 0]
    delta2 = delta[..., 1]
    dist2 = np.square(delta1) + np.square(delta2)
    return np.stack((dist2, delta1, delta2), axis=-1)


@lru_cache(maxsize=16)
def _cached_rel_pos(grid_size: tuple[int, int]) -> np.ndarray:
    return _build_rel_pos(grid_size)


class SelfAttention(eqx.Module):
    """Self-attention with optional causality and grouped query attention.

    :ivar d_model: Model width.
    :ivar n_heads: Number of query heads.
    :ivar n_kv_heads: Number of key/value heads for GQA.
    :ivar attn_dropout: Attention dropout probability.
    :ivar resid_dropout: Residual dropout probability.
    :ivar is_causal: Whether to apply a causal attention mask.
    :ivar dtype: Compute dtype.
    :ivar param_dtype: Parameter dtype.
    :ivar q_proj: Query projection.
    :ivar k_proj: Key projection.
    :ivar v_proj: Value projection.
    :ivar o_proj: Output projection.
    :ivar attn_dropout_layer: Dropout layer for attention weights.
    :ivar resid_dropout_layer: Dropout layer for residual projections.
    """

    d_model: int
    n_heads: int
    n_kv_heads: int
    attn_dropout: float
    resid_dropout: float
    is_causal: bool
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    attn_implementation: str | None = eqx.field(static=True)
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    attn_dropout_layer: eqx.nn.Dropout
    resid_dropout_layer: eqx.nn.Dropout

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        *,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        is_causal: bool = True,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        qkv_kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        o_kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        attn_implementation: str | None = None,
        key: Array,
    ) -> None:
        """Initialize attention parameters.

        :param d_model: Model width.
        :param n_heads: Number of query heads.
        :param n_kv_heads: Number of key/value heads for GQA.
        :param attn_dropout: Attention dropout probability.
        :param resid_dropout: Residual dropout probability.
        :param is_causal: Whether to apply a causal attention mask.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param qkv_kernel_init: Initializer for Q/K/V projections.
        :param o_kernel_init: Initializer for output projection.
        :param key: PRNG key for parameter initialization.
        """
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be > 0")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        head_dim = d_model // n_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if not (1 <= n_kv_heads <= n_heads):
            raise ValueError("n_kv_heads must be in [1, n_heads]")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads (for GQA)")

        q_key, k_key, v_key, o_key = jax.random.split(key, 4)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.is_causal = is_causal
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.attn_implementation = attn_implementation

        self.q_proj = Linear(
            in_features=d_model,
            out_features=n_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=qkv_kernel_init,
            key=q_key,
        )
        self.k_proj = Linear(
            in_features=d_model,
            out_features=n_kv_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=qkv_kernel_init,
            key=k_key,
        )
        self.v_proj = Linear(
            in_features=d_model,
            out_features=n_kv_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=qkv_kernel_init,
            key=v_key,
        )
        self.o_proj = Linear(
            in_features=n_heads * head_dim,
            out_features=d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=o_kernel_init,
            key=o_key,
        )
        self.attn_dropout_layer = eqx.nn.Dropout(attn_dropout)
        self.resid_dropout_layer = eqx.nn.Dropout(resid_dropout)

    def __call__(self, x: Array, *, train: bool, key: Array | None) -> Array:
        """Compute self-attention.

        :param x: Input tensor of shape (B, T, d_model).
        :param train: Whether to enable dropout.
        :param key: PRNG key for dropout.
        :returns: Output tensor of shape (B, T, d_model).
        :raises ValueError: If dropout is enabled without a PRNG key.
        """
        if train is True and (self.attn_dropout > 0.0 or self.resid_dropout > 0.0) and key is None:
            raise ValueError("dropout key must be provided when training with dropout")

        bsz, seqlen, _ = x.shape
        head_dim = self.d_model // self.n_heads

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        use_fast_attn = self.attn_implementation is not None and self.attn_dropout == 0.0
        if use_fast_attn:
            q = q.reshape((bsz, seqlen, self.n_heads, head_dim))
            k = k.reshape((bsz, seqlen, self.n_kv_heads, head_dim))
            v = v.reshape((bsz, seqlen, self.n_kv_heads, head_dim))
            y = jax.nn.dot_product_attention(
                q,
                k,
                v,
                is_causal=self.is_causal,
                implementation=self.attn_implementation,
            )
            y = y.reshape((bsz, seqlen, self.n_heads * head_dim))
            y = self.o_proj(y)
            if self.resid_dropout > 0.0 and key is not None:
                key, resid_key = jax.random.split(key)
                y = self.resid_dropout_layer(y, key=resid_key, inference=train is False)
            return y

        q = q.reshape((bsz, seqlen, self.n_heads, head_dim)).transpose((0, 2, 1, 3))
        k = k.reshape((bsz, seqlen, self.n_kv_heads, head_dim)).transpose((0, 2, 1, 3))
        v = v.reshape((bsz, seqlen, self.n_kv_heads, head_dim)).transpose((0, 2, 1, 3))

        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = jnp.repeat(k, repeats=repeat, axis=1)
            v = jnp.repeat(v, repeats=repeat, axis=1)

        qf = q.astype(jnp.float32)
        kf = k.astype(jnp.float32)
        vf = v.astype(v.dtype)

        att = jnp.matmul(qf, jnp.swapaxes(kf, -2, -1)) / math.sqrt(head_dim)

        if self.is_causal is True:
            mask = jnp.triu(jnp.ones((seqlen, seqlen), dtype=bool), k=1)
            att = jnp.where(mask[None, None, :, :], -jnp.inf, att)

        att = jax.nn.softmax(att, axis=-1)
        if self.attn_dropout > 0.0 and key is not None:
            key, attn_key = jax.random.split(key)
            att = self.attn_dropout_layer(att, key=attn_key, inference=train is False)

        y = jnp.matmul(att, vf).astype(q.dtype)
        y = y.transpose((0, 2, 1, 3)).reshape((bsz, seqlen, self.n_heads * head_dim))
        y = self.o_proj(y)
        if self.resid_dropout > 0.0 and key is not None:
            key, resid_key = jax.random.split(key)
            y = self.resid_dropout_layer(y, key=resid_key, inference=train is False)
        return y


def _pope_frequencies(head_dim: int, *, base: float) -> jnp.ndarray:
    """Compute PoPE angular frequencies.

    :param head_dim: Per-head dimension.
    :param base: Base for frequency computation.
    :returns: Frequency vector of shape (head_dim,).
    """
    if head_dim <= 0:
        raise ValueError("head_dim must be > 0")
    if base <= 1.0:
        raise ValueError("base must be > 1.0")
    idx = jnp.arange(head_dim, dtype=jnp.float32)
    return jnp.exp(-jnp.log(jnp.asarray(base, dtype=jnp.float32)) * idx / head_dim)


class PoPESelfAttention(eqx.Module):
    """Self-attention with PoPE positional encoding.

    :ivar d_model: Model width.
    :ivar n_heads: Number of query heads.
    :ivar n_kv_heads: Number of key/value heads for GQA.
    :ivar attn_dropout: Attention dropout probability.
    :ivar resid_dropout: Residual dropout probability.
    :ivar is_causal: Whether to apply a causal attention mask.
    :ivar dtype: Compute dtype.
    :ivar param_dtype: Parameter dtype.
    :ivar q_proj: Query projection.
    :ivar k_proj: Key projection.
    :ivar v_proj: Value projection.
    :ivar o_proj: Output projection.
    :ivar attn_dropout_layer: Dropout layer for attention weights.
    :ivar resid_dropout_layer: Dropout layer for residual projections.
    :ivar delta: Learnable PoPE phase bias of shape (n_heads, head_dim).
    """

    d_model: int
    n_heads: int
    n_kv_heads: int
    attn_dropout: float
    resid_dropout: float
    is_causal: bool
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    base: float = eqx.field(static=True)
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    attn_dropout_layer: eqx.nn.Dropout
    resid_dropout_layer: eqx.nn.Dropout
    delta: Array

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        *,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        is_causal: bool = True,
        base: float = 10000.0,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        qkv_kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        o_kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        key: Array,
    ) -> None:
        """Initialize PoPE attention parameters.

        :param d_model: Model width.
        :param n_heads: Number of query heads.
        :param n_kv_heads: Number of key/value heads for GQA.
        :param attn_dropout: Attention dropout probability.
        :param resid_dropout: Residual dropout probability.
        :param is_causal: Whether to apply a causal attention mask.
        :param base: Base for the PoPE frequency schedule.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param qkv_kernel_init: Initializer for Q/K/V projections.
        :param o_kernel_init: Initializer for output projection.
        :param key: PRNG key for parameter initialization.
        """
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be > 0")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not (1 <= n_kv_heads <= n_heads):
            raise ValueError("n_kv_heads must be in [1, n_heads]")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads (for GQA)")
        if base <= 1.0:
            raise ValueError("base must be > 1.0")

        head_dim = d_model // n_heads
        q_key, k_key, v_key, o_key, delta_key = jax.random.split(key, 5)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.is_causal = is_causal
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.base = base

        self.q_proj = Linear(
            in_features=d_model,
            out_features=n_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=qkv_kernel_init,
            key=q_key,
        )
        self.k_proj = Linear(
            in_features=d_model,
            out_features=n_kv_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=qkv_kernel_init,
            key=k_key,
        )
        self.v_proj = Linear(
            in_features=d_model,
            out_features=n_kv_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=qkv_kernel_init,
            key=v_key,
        )
        self.o_proj = Linear(
            in_features=n_heads * head_dim,
            out_features=d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=o_kernel_init,
            key=o_key,
        )
        self.attn_dropout_layer = eqx.nn.Dropout(attn_dropout)
        self.resid_dropout_layer = eqx.nn.Dropout(resid_dropout)
        self.delta = jnp.zeros((n_heads, head_dim), dtype=param_dtype)

    def __call__(
        self,
        x: Array,
        *,
        train: bool,
        key: Array | None,
        attention_mask: Array | None = None,
    ) -> Array:
        """Compute PoPE self-attention.

        :param x: Input tensor of shape (B, T, d_model).
        :param train: Whether to enable dropout.
        :param key: PRNG key for dropout.
        :param attention_mask: Optional mask of shape (B, T) for valid tokens.
        :returns: Output tensor of shape (B, T, d_model).
        :raises ValueError: If dropout is enabled without a PRNG key.
        """
        if train is True and (self.attn_dropout > 0.0 or self.resid_dropout > 0.0) and key is None:
            raise ValueError("dropout key must be provided when training with dropout")
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("attention_mask must have shape (B, T)")

        bsz, seqlen, _ = x.shape
        head_dim = self.d_model // self.n_heads

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape((bsz, seqlen, self.n_heads, head_dim)).transpose((0, 2, 1, 3))
        k = k.reshape((bsz, seqlen, self.n_kv_heads, head_dim)).transpose((0, 2, 1, 3))
        v = v.reshape((bsz, seqlen, self.n_kv_heads, head_dim)).transpose((0, 2, 1, 3))

        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = jnp.repeat(k, repeats=repeat, axis=1)
            v = jnp.repeat(v, repeats=repeat, axis=1)

        qf = q.astype(jnp.float32)
        kf = k.astype(jnp.float32)
        vf = v.astype(v.dtype)

        positions = jnp.arange(seqlen, dtype=jnp.float32)
        freqs = _pope_frequencies(head_dim, base=self.base)
        angles = positions[:, None] * freqs[None, :]

        q_mag = jax.nn.softplus(qf)
        q_cos = jnp.cos(angles)[None, None, :, :]
        q_sin = jnp.sin(angles)[None, None, :, :]
        q_real = q_mag * q_cos
        q_imag = q_mag * q_sin

        delta = jnp.clip(self.delta.astype(jnp.float32), a_min=-2.0 * jnp.pi, a_max=0.0)
        k_angles = angles[None, None, :, :] + delta[None, :, None, :]
        k_mag = jax.nn.softplus(kf)
        k_real = k_mag * jnp.cos(k_angles)
        k_imag = k_mag * jnp.sin(k_angles)

        att = jnp.matmul(q_real, jnp.swapaxes(k_real, -2, -1))
        att = att + jnp.matmul(q_imag, jnp.swapaxes(k_imag, -2, -1))
        att = att / math.sqrt(head_dim)

        if self.is_causal is True:
            mask = jnp.triu(jnp.ones((seqlen, seqlen), dtype=bool), k=1)
            att = jnp.where(mask[None, None, :, :], -jnp.inf, att)
        if attention_mask is not None:
            mask = attention_mask.astype(bool)
            att = jnp.where(mask[:, None, None, :], att, -jnp.inf)

        att = jax.nn.softmax(att, axis=-1)
        if self.attn_dropout > 0.0 and key is not None:
            key, attn_key = jax.random.split(key)
            att = self.attn_dropout_layer(att, key=attn_key, inference=train is False)

        y = jnp.matmul(att, vf).astype(q.dtype)
        if attention_mask is not None:
            mask = attention_mask.astype(y.dtype)
            y = y * mask[:, None, :, None]
        y = y.transpose((0, 2, 1, 3)).reshape((bsz, seqlen, self.n_heads * head_dim))
        y = self.o_proj(y)
        if self.resid_dropout > 0.0 and key is not None:
            key, resid_key = jax.random.split(key)
            y = self.resid_dropout_layer(y, key=resid_key, inference=train is False)
        return y


class GatedPositionalSelfAttention(eqx.Module):
    """Gated positional self-attention (GPSA) with convolutional initialization.

    :ivar d_model: Model width.
    :ivar n_heads: Number of attention heads.
    :ivar grid_size: Patch grid size as (height, width).
    :ivar attn_dropout: Attention dropout probability.
    :ivar resid_dropout: Residual dropout probability.
    :ivar dtype: Compute dtype.
    :ivar param_dtype: Parameter dtype.
    :ivar q_proj: Query projection.
    :ivar k_proj: Key projection.
    :ivar v_proj: Value projection.
    :ivar o_proj: Output projection.
    :ivar v_pos: Positional embedding vectors per head.
    :ivar gating_param: Learnable gating parameters (lambda) per head.
    :ivar attn_dropout_layer: Dropout layer for attention weights.
    :ivar resid_dropout_layer: Dropout layer for residual projections.
    """

    d_model: int
    n_heads: int
    grid_size: tuple[int, int]
    attn_dropout: float
    resid_dropout: float
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    attn_implementation: str | None = eqx.field(static=True)
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    v_pos: Array
    gating_param: Array
    attn_dropout_layer: eqx.nn.Dropout
    resid_dropout_layer: eqx.nn.Dropout

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        grid_size: tuple[int, int],
        *,
        locality_strength: float | Array = 1.0,
        attention_centers: Array | None = None,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        qkv_kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        o_kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        attn_implementation: str | None = None,
        key: Array,
    ) -> None:
        """Initialize GPSA parameters.

        :param d_model: Model width.
        :param n_heads: Number of attention heads.
        :param grid_size: Patch grid size as (height, width).
        :param locality_strength: Initial locality strength (alpha).
        :param attention_centers: Optional attention centers of shape (n_heads, 2).
        :param attn_dropout: Attention dropout probability.
        :param resid_dropout: Residual dropout probability.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param qkv_kernel_init: Initializer for Q/K/V projections.
        :param o_kernel_init: Initializer for output projection.
        :param key: PRNG key for parameter initialization.
        """
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if len(grid_size) != 2:
            raise ValueError("grid_size must be a tuple of (height, width)")
        if grid_size[0] <= 0 or grid_size[1] <= 0:
            raise ValueError("grid_size entries must be > 0")
        if isinstance(locality_strength, (int, float)) and locality_strength <= 0.0:
            raise ValueError("locality_strength must be > 0")

        head_dim = d_model // n_heads
        q_key, k_key, v_key, o_key = jax.random.split(key, 4)

        self.d_model = d_model
        self.n_heads = n_heads
        self.grid_size = grid_size
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.attn_implementation = attn_implementation

        self.q_proj = Linear(
            in_features=d_model,
            out_features=n_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=qkv_kernel_init,
            key=q_key,
        )
        self.k_proj = Linear(
            in_features=d_model,
            out_features=n_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=qkv_kernel_init,
            key=k_key,
        )
        self.v_proj = Linear(
            in_features=d_model,
            out_features=n_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=qkv_kernel_init,
            key=v_key,
        )
        self.o_proj = Linear(
            in_features=n_heads * head_dim,
            out_features=d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=o_kernel_init,
            key=o_key,
        )

        if attention_centers is None:
            grid_dim = int(math.sqrt(n_heads))
            if grid_dim * grid_dim != n_heads:
                raise ValueError("n_heads must be a perfect square when attention_centers is not provided")
            if grid_dim == 1:
                centers = jnp.zeros((1, 2), dtype=param_dtype)
            else:
                centers_1d = jnp.linspace(-1.0, 1.0, grid_dim).astype(param_dtype)
                centers_y, centers_x = jnp.meshgrid(centers_1d, centers_1d, indexing="ij")
                centers = jnp.stack((centers_y, centers_x), axis=-1).reshape((n_heads, 2))
        else:
            centers = jnp.asarray(attention_centers, dtype=param_dtype)
            if centers.shape != (n_heads, 2):
                raise ValueError("attention_centers must have shape (n_heads, 2)")

        if isinstance(locality_strength, (int, float)):
            alpha = jnp.full((n_heads,), float(locality_strength), dtype=param_dtype)
        else:
            alpha = jnp.asarray(locality_strength, dtype=param_dtype)
            if alpha.shape != (n_heads,):
                raise ValueError("locality_strength must be a scalar or shape (n_heads,)")

        ones = jnp.ones((n_heads,), dtype=param_dtype)
        self.v_pos = -alpha[:, None] * jnp.stack(
            (ones, -2.0 * centers[:, 0], -2.0 * centers[:, 1]),
            axis=-1,
        )
        self.gating_param = jnp.full((n_heads,), 1.0, dtype=param_dtype)
        self.attn_dropout_layer = eqx.nn.Dropout(attn_dropout)
        self.resid_dropout_layer = eqx.nn.Dropout(resid_dropout)

    def __call__(self, x: Array, *, train: bool, key: Array | None) -> Array:
        """Compute gated positional self-attention.

        :param x: Input tensor of shape (B, T, d_model).
        :param train: Whether to enable dropout.
        :param key: PRNG key for dropout.
        :returns: Output tensor of shape (B, T, d_model).
        :raises ValueError: If dropout is enabled without a PRNG key.
        """
        if train is True and (self.attn_dropout > 0.0 or self.resid_dropout > 0.0) and key is None:
            raise ValueError("dropout key must be provided when training with dropout")

        bsz, seqlen, _ = x.shape
        expected_seqlen = self.grid_size[0] * self.grid_size[1]
        if seqlen == expected_seqlen:
            rel_pos = _cached_rel_pos(self.grid_size)
        else:
            grid_dim = int(math.sqrt(seqlen))
            if grid_dim * grid_dim != seqlen:
                raise ValueError("sequence length must be a perfect square for GPSA")
            rel_pos = _cached_rel_pos((grid_dim, grid_dim))

        head_dim = self.d_model // self.n_heads

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        use_fast_attn = self.attn_implementation is not None and self.attn_dropout == 0.0
        if use_fast_attn:
            q = q.reshape((bsz, seqlen, self.n_heads, head_dim))
            k = k.reshape((bsz, seqlen, self.n_heads, head_dim))
            v = v.reshape((bsz, seqlen, self.n_heads, head_dim))
            content_out = jax.nn.dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                implementation=self.attn_implementation,
            )

            pos_scores = jnp.einsum(
                "hd,ijd->hij",
                self.v_pos.astype(jnp.float32),
                jnp.asarray(rel_pos),
            )
            pos_attn = jax.nn.softmax(pos_scores, axis=-1)[None, ...]
            vf = v.transpose((0, 2, 1, 3)).astype(jnp.float32)
            pos_out = jnp.matmul(pos_attn, vf).transpose((0, 2, 1, 3))

            gating = jax.nn.sigmoid(self.gating_param.astype(jnp.float32)).astype(content_out.dtype)
            gating = gating[None, None, :, None]
            y = (1.0 - gating) * content_out + gating * pos_out.astype(content_out.dtype)
            y = y.reshape((bsz, seqlen, self.n_heads * head_dim))
            y = self.o_proj(y)
            if self.resid_dropout > 0.0 and key is not None:
                key, resid_key = jax.random.split(key)
                y = self.resid_dropout_layer(y, key=resid_key, inference=train is False)
            return y

        q = q.reshape((bsz, seqlen, self.n_heads, head_dim)).transpose((0, 2, 1, 3))
        k = k.reshape((bsz, seqlen, self.n_heads, head_dim)).transpose((0, 2, 1, 3))
        v = v.reshape((bsz, seqlen, self.n_heads, head_dim)).transpose((0, 2, 1, 3))

        qf = q.astype(jnp.float32)
        kf = k.astype(jnp.float32)
        vf = v.astype(jnp.float32)

        content_scores = jnp.matmul(qf, jnp.swapaxes(kf, -2, -1)) / math.sqrt(head_dim)
        content_attn = jax.nn.softmax(content_scores, axis=-1)

        pos_scores = jnp.einsum(
            "hd,ijd->hij",
            self.v_pos.astype(jnp.float32),
            jnp.asarray(rel_pos),
        )
        pos_attn = jax.nn.softmax(pos_scores, axis=-1)[None, ...]

        gating = jax.nn.sigmoid(self.gating_param.astype(jnp.float32))[None, :, None, None]
        attn = (1.0 - gating) * content_attn + gating * pos_attn

        if self.attn_dropout > 0.0 and key is not None:
            key, attn_key = jax.random.split(key)
            attn = self.attn_dropout_layer(attn, key=attn_key, inference=train is False)

        y = jnp.matmul(attn, vf).astype(q.dtype)
        y = y.transpose((0, 2, 1, 3)).reshape((bsz, seqlen, self.n_heads * head_dim))
        y = self.o_proj(y)
        if self.resid_dropout > 0.0 and key is not None:
            key, resid_key = jax.random.split(key)
            y = self.resid_dropout_layer(y, key=resid_key, inference=train is False)
        return y
