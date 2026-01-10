from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

Initializer = Callable[[Array, tuple[int, ...], jnp.dtype], Array]


def truncated_normal_init(std: float) -> Initializer:
    """Build a truncated normal initializer (±2σ) with the provided std.

    :param std: Standard deviation for the distribution.
    :returns: Initializer function for model parameters.
    """
    if std <= 0.0:
        raise ValueError("std must be > 0")

    def _init(key: Array, shape: tuple[int, ...], dtype: jnp.dtype) -> Array:
        values = jax.random.truncated_normal(key, lower=-2.0, upper=2.0, shape=shape, dtype=dtype)
        return values * jnp.asarray(std, dtype=dtype)

    return _init
