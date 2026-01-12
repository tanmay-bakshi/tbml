import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from tbml.nn.init import Initializer
from tbml.nn.linear import Linear


class RMSNorm(eqx.Module):
    """RMSNorm (no mean subtraction), with a single learnable scale vector.

    :ivar dim: Size of the last dimension to normalize.
    :ivar eps: Epsilon added to the RMS denominator.
    :ivar dtype: Compute dtype.
    :ivar param_dtype: Parameter dtype.
    :ivar weight: Learnable scale vector of shape (dim,).
    """

    dim: int
    eps: float
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    weight: Array

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        """Initialize RMSNorm parameters.

        :param dim: Size of the last dimension to normalize.
        :param eps: Epsilon added to the RMS denominator.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        """
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")

        self.dim = dim
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.weight = jnp.ones((dim,), dtype=param_dtype)

    def __call__(self, x: Array) -> Array:
        """Apply RMSNorm over the last dimension.

        :param x: Input tensor of shape (..., dim).
        :returns: Normalized tensor of the same shape as ``x``.
        """
        rms = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        scale = jax.lax.rsqrt(rms + self.eps).astype(x.dtype)
        return (x * scale) * self.weight.astype(x.dtype)


class AdaptiveRMSNorm(eqx.Module):
    """Adaptive RMSNorm conditioned on an external embedding.

    :ivar dim: Size of the last dimension to normalize.
    :ivar cond_dim: Size of the conditioning embedding.
    :ivar dtype: Compute dtype.
    :ivar param_dtype: Parameter dtype.
    :ivar norm: Base RMSNorm module.
    :ivar cond_proj: Linear projection producing scale and shift vectors.
    """

    dim: int
    cond_dim: int
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    norm: RMSNorm
    cond_proj: Linear

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        *,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        kernel_init: Initializer = jax.nn.initializers.zeros,
        key: Array,
    ) -> None:
        """Initialize adaptive RMSNorm parameters.

        :param dim: Size of the last dimension to normalize.
        :param cond_dim: Size of the conditioning embedding.
        :param eps: Epsilon added to the RMS denominator.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param kernel_init: Initializer for the conditioning projection.
        :param key: PRNG key for parameter initialization.
        """
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if cond_dim <= 0:
            raise ValueError("cond_dim must be > 0")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")

        self.dim = dim
        self.cond_dim = cond_dim
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.norm = RMSNorm(dim, eps=eps, dtype=dtype, param_dtype=param_dtype)
        self.cond_proj = Linear(
            in_features=cond_dim,
            out_features=2 * dim,
            use_bias=True,
            bias_value=0.0,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            key=key,
        )

    def __call__(self, x: Array, cond: Array) -> Array:
        """Apply adaptive RMSNorm.

        :param x: Input tensor of shape (B, T, dim).
        :param cond: Conditioning tensor of shape (B, cond_dim).
        :returns: Normalized tensor of shape (B, T, dim).
        """
        if x.ndim != 3:
            raise ValueError("x must have shape (B, T, dim)")
        if cond.ndim != 2:
            raise ValueError("cond must have shape (B, cond_dim)")
        if x.shape[0] != cond.shape[0]:
            raise ValueError("x and cond must have the same batch size")
        if x.shape[-1] != self.dim:
            raise ValueError("x last dimension must match dim")
        if cond.shape[-1] != self.cond_dim:
            raise ValueError("cond last dimension must match cond_dim")

        normed = self.norm(x)
        scale_shift = self.cond_proj(cond)
        scale, shift = jnp.split(scale_shift, 2, axis=-1)
        scale = scale[:, None, :]
        shift = shift[:, None, :]
        return normed * (1.0 + scale.astype(normed.dtype)) + shift.astype(normed.dtype)
