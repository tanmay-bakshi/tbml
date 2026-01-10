import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


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
