import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from tbml.nn.init import Initializer


class Linear(eqx.Module):
    """Linear projection with an optional bias.

    :ivar weight: Weight matrix of shape (out_features, in_features).
    :ivar bias: Optional bias vector of shape (out_features,).
    :ivar in_features: Input feature dimension.
    :ivar out_features: Output feature dimension.
    :ivar use_bias: Whether the bias is enabled.
    :ivar dtype: Compute dtype.
    :ivar param_dtype: Parameter dtype.
    """

    weight: Array
    bias: Array | None
    in_features: int
    out_features: int
    use_bias: bool
    dtype: jnp.dtype
    param_dtype: jnp.dtype

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: Array,
        use_bias: bool = False,
        bias_value: float | None = None,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
    ) -> None:
        """Initialize the projection parameters.

        :param in_features: Input feature dimension.
        :param out_features: Output feature dimension.
        :param key: PRNG key for parameter initialization.
        :param use_bias: Whether to include a bias term.
        :param bias_value: Optional constant value for bias initialization.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param kernel_init: Initializer for the weight matrix.
        """
        if in_features <= 0:
            raise ValueError("in_features must be > 0")
        if out_features <= 0:
            raise ValueError("out_features must be > 0")

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.weight = kernel_init(key, (out_features, in_features), param_dtype)
        if use_bias is True:
            if bias_value is None:
                self.bias = jnp.zeros((out_features,), dtype=param_dtype)
            else:
                self.bias = jnp.full((out_features,), fill_value=bias_value, dtype=param_dtype)
        else:
            self.bias = None

    def __call__(self, x: Array) -> Array:
        """Apply the linear projection.

        :param x: Input tensor of shape (..., in_features).
        :returns: Output tensor of shape (..., out_features).
        """
        weight = self.weight.astype(self.dtype)
        output = jnp.einsum("...i,oi->...o", x, weight)
        if self.bias is not None:
            output = output + self.bias.astype(self.dtype)
        return output
