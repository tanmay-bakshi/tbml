import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from tbml.nn import Linear
from tbml.nn.init import Initializer


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

        q = q.reshape((bsz, seqlen, self.n_heads, head_dim)).transpose((0, 2, 1, 3))
        k = k.reshape((bsz, seqlen, self.n_kv_heads, head_dim)).transpose((0, 2, 1, 3))
        v = v.reshape((bsz, seqlen, self.n_kv_heads, head_dim)).transpose((0, 2, 1, 3))

        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = jnp.repeat(k, repeats=repeat, axis=1)
            v = jnp.repeat(v, repeats=repeat, axis=1)

        qf = q.astype(jnp.float32)
        kf = k.astype(jnp.float32)
        vf = v.astype(jnp.float32)

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
