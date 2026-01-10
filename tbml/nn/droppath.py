import equinox as eqx
import jax
from jaxtyping import Array


class DropPath(eqx.Module):
    """Residual DropPath (stochastic depth) regularizer.

    :ivar drop_prob: Probability of dropping a residual path.
    """

    drop_prob: float

    def __init__(self, drop_prob: float) -> None:
        """Initialize DropPath parameters.

        :param drop_prob: Probability of dropping a residual path.
        """
        if drop_prob < 0.0:
            raise ValueError("drop_prob must be >= 0")
        if drop_prob >= 1.0:
            raise ValueError("drop_prob must be < 1")

        self.drop_prob = drop_prob

    def __call__(self, x: Array, *, train: bool, key: Array | None) -> Array:
        """Apply DropPath to a residual branch.

        :param x: Input tensor of shape (B, ...).
        :param train: Whether to enable stochastic depth.
        :param key: PRNG key for DropPath.
        :returns: Tensor of the same shape as ``x``.
        :raises ValueError: If DropPath is enabled without a PRNG key.
        """
        if train is False or self.drop_prob == 0.0:
            return x
        if key is None:
            raise ValueError("drop path key must be provided when training with drop_prob > 0")
        if x.ndim < 1:
            raise ValueError("drop path expects a batch dimension")

        keep_prob = 1.0 - self.drop_prob
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = jax.random.bernoulli(key, p=keep_prob, shape=mask_shape)
        mask = mask.astype(x.dtype)
        return (x * mask) / keep_prob
