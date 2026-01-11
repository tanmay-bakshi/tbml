import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from tbml.nn.init import Initializer
from tbml.nn.linear import Linear


class GELUFeedForward(eqx.Module):
    """GELU MLP: W_down(GELU(W_up x)).

    :ivar d_model: Model width.
    :ivar hidden_dim: Hidden dimension size for the feed-forward expansion.
    :ivar resid_dropout: Residual dropout probability.
    :ivar dtype: Compute dtype.
    :ivar param_dtype: Parameter dtype.
    :ivar up_proj: Up projection.
    :ivar down_proj: Down projection.
    :ivar resid_dropout_layer: Dropout layer for residual outputs.
    """

    d_model: int
    hidden_dim: int
    resid_dropout: float
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    up_proj: Linear
    down_proj: Linear
    resid_dropout_layer: eqx.nn.Dropout

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        *,
        resid_dropout: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        up_kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        down_kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        key: Array,
    ) -> None:
        """Initialize feed-forward projections.

        :param d_model: Model width.
        :param hidden_dim: Hidden dimension size for the feed-forward expansion.
        :param resid_dropout: Residual dropout probability.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param up_kernel_init: Initializer for the up projection.
        :param down_kernel_init: Initializer for down projection.
        :param key: PRNG key for parameter initialization.
        """
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        up_key, down_key = jax.random.split(key, 2)

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.resid_dropout = resid_dropout
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.up_proj = Linear(
            in_features=d_model,
            out_features=hidden_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=up_kernel_init,
            key=up_key,
        )
        self.down_proj = Linear(
            in_features=hidden_dim,
            out_features=d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=down_kernel_init,
            key=down_key,
        )
        self.resid_dropout_layer = eqx.nn.Dropout(resid_dropout)

    def __call__(self, x: Array, *, train: bool, key: Array | None) -> Array:
        """Compute GELU feed-forward transformation.

        :param x: Input tensor of shape (B, T, d_model).
        :param train: Whether to enable dropout.
        :param key: PRNG key for dropout.
        :returns: Output tensor of shape (B, T, d_model).
        :raises ValueError: If dropout is enabled without a PRNG key.
        """
        hidden = self.up_proj(x)
        hidden = jax.nn.gelu(hidden)
        out = self.down_proj(hidden)
        if self.resid_dropout > 0.0 and key is not None:
            out = self.resid_dropout_layer(out, key=key, inference=train is False)
        return out
