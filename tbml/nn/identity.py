import equinox as eqx
from jaxtyping import Array


class Identity(eqx.Module):
    """Identity module that returns the input unchanged."""

    def __call__(self, x: Array) -> Array:
        """Return the input unchanged.

        :param x: Input tensor.
        :returns: The same tensor.
        """
        return x
