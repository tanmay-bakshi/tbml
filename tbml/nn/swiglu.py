import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from tbml.nn.linear import Linear
from tbml.nn.init import Initializer


class SwiGLUFeedForward(eqx.Module):
    """SwiGLU MLP: (SiLU(W_gate x) ⊙ (W_up x)) W_down.

    :ivar d_model: Model width.
    :ivar d_out: Output width after the down projection.
    :ivar hidden_dim: Hidden dimension size for the feed-forward expansion.
    :ivar resid_dropout: Residual dropout probability.
    :ivar dtype: Compute dtype.
    :ivar param_dtype: Parameter dtype.
    :ivar gate_proj: Gate projection.
    :ivar up_proj: Up projection.
    :ivar down_proj: Down projection.
    :ivar resid_dropout_layer: Dropout layer for residual outputs.
    """

    d_model: int
    d_out: int
    hidden_dim: int
    resid_dropout: float
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear
    resid_dropout_layer: eqx.nn.Dropout

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        d_out: int | None = None,
        *,
        resid_dropout: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        gate_up_kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        down_kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        key: Array,
    ) -> None:
        """Initialize feed-forward projections.

        :param d_model: Model width.
        :param hidden_dim: Hidden dimension size for the feed-forward expansion.
        :param d_out: Output width after the down projection.
        :param resid_dropout: Residual dropout probability.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param gate_up_kernel_init: Initializer for the up projection.
        :param down_kernel_init: Initializer for down projection.
        :param key: PRNG key for parameter initialization.
        """
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if d_out is not None and d_out <= 0:
            raise ValueError("d_out must be > 0 when set")

        out_dim = d_model if d_out is None else d_out

        gate_key, up_key, down_key = jax.random.split(key, 3)

        self.d_model = d_model
        self.d_out = out_dim
        self.hidden_dim = hidden_dim
        self.resid_dropout = resid_dropout
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.gate_proj = Linear(
            in_features=d_model,
            out_features=hidden_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=gate_up_kernel_init,
            key=gate_key,
        )
        self.up_proj = Linear(
            in_features=d_model,
            out_features=hidden_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=gate_up_kernel_init,
            key=up_key,
        )
        self.down_proj = Linear(
            in_features=hidden_dim,
            out_features=out_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=down_kernel_init,
            key=down_key,
        )
        self.resid_dropout_layer = eqx.nn.Dropout(resid_dropout)

    def __call__(self, x: Array, *, train: bool, key: Array | None) -> Array:
        """Compute SwiGLU feed-forward transformation.

        :param x: Input tensor of shape (B, T, d_model).
        :param train: Whether to enable dropout.
        :param key: PRNG key for dropout.
        :returns: Output tensor of shape (B, T, d_out).
        :raises ValueError: If dropout is enabled without a PRNG key.
        """
        gate_weight = self.gate_proj.weight.astype(self.dtype)
        up_weight = self.up_proj.weight.astype(self.dtype)
        fused_weight = jnp.concatenate([gate_weight, up_weight], axis=0)
        proj = jnp.einsum("...i,oi->...o", x, fused_weight)
        if self.gate_proj.bias is not None and self.up_proj.bias is not None:
            fused_bias = jnp.concatenate(
                [self.gate_proj.bias.astype(self.dtype), self.up_proj.bias.astype(self.dtype)],
                axis=0,
            )
            proj = proj + fused_bias
        gate, up = jnp.split(proj, 2, axis=-1)
        gated = jax.nn.silu(gate) * up
        out = self.down_proj(gated)
        if self.resid_dropout > 0.0 and key is not None:
            out = self.resid_dropout_layer(out, key=key, inference=train is False)
        return out
