import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from tbml.nn.init import Initializer
from tbml.nn.linear import Linear
from tbml.nn.rmsnorm import RMSNorm


class LSTMCell(eqx.Module):
    """LSTM cell with RMSNorm on gate projections and cell state."""

    input_dim: int
    hidden_dim: int
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    proj_h: Linear
    proj_x: Linear
    bias: Array
    norm_h: RMSNorm
    norm_x: RMSNorm
    norm_c: RMSNorm

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        key: Array,
    ) -> None:
        """Initialize the LSTM cell.

        :param input_dim: Input feature dimension.
        :param hidden_dim: Hidden state dimension.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param kernel_init: Initializer for the projection weights.
        :param key: PRNG key for parameter initialization.
        """
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.param_dtype = param_dtype
        proj_h_key, proj_x_key, norm_h_key, norm_x_key, norm_c_key = jax.random.split(key, 5)
        self.proj_h = Linear(
            in_features=hidden_dim,
            out_features=4 * hidden_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            key=proj_h_key,
        )
        self.proj_x = Linear(
            in_features=input_dim,
            out_features=4 * hidden_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            key=proj_x_key,
        )
        self.bias = jnp.zeros((4 * hidden_dim,), dtype=param_dtype)
        self.norm_h = RMSNorm(4 * hidden_dim, dtype=dtype, param_dtype=param_dtype)
        self.norm_x = RMSNorm(4 * hidden_dim, dtype=dtype, param_dtype=param_dtype)
        self.norm_c = RMSNorm(hidden_dim, dtype=dtype, param_dtype=param_dtype)
        _ = norm_h_key
        _ = norm_x_key
        _ = norm_c_key

    def __call__(
        self,
        x: Array,
        h: Array,
        c: Array,
    ) -> tuple[Array, Array]:
        """Apply the LSTM cell.

        :param x: Input tensor of shape (B, input_dim).
        :param h: Hidden state of shape (B, hidden_dim).
        :param c: Cell state of shape (B, hidden_dim).
        :returns: Tuple of (new hidden, new cell).
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError("x last dimension must match input_dim")
        if h.shape[-1] != self.hidden_dim:
            raise ValueError("h last dimension must match hidden_dim")
        if c.shape[-1] != self.hidden_dim:
            raise ValueError("c last dimension must match hidden_dim")

        proj_h = self.proj_h(h)
        proj_x = self.proj_x(x)
        gates = self.norm_h(proj_h) + self.norm_x(proj_x) + self.bias.astype(self.dtype)
        f_gate, i_gate, o_gate, g_gate = jnp.split(gates, 4, axis=-1)
        f_gate = jax.nn.sigmoid(f_gate)
        i_gate = jax.nn.sigmoid(i_gate)
        o_gate = jax.nn.sigmoid(o_gate)
        g_gate = jnp.tanh(g_gate)
        new_c = f_gate * c + i_gate * g_gate
        norm_c = self.norm_c(new_c)
        new_h = o_gate * jnp.tanh(norm_c)
        return new_h, new_c


class LSTMLayer(eqx.Module):
    """Single LSTM layer."""

    cell: LSTMCell

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        key: Array,
    ) -> None:
        """Initialize the LSTM layer.

        :param input_dim: Input feature dimension.
        :param hidden_dim: Hidden state dimension.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param kernel_init: Initializer for the projection weights.
        :param key: PRNG key for parameter initialization.
        """
        cell_key, norm_key = jax.random.split(key, 2)
        self.cell = LSTMCell(
            input_dim,
            hidden_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            key=cell_key,
        )
        _ = norm_key

    def __call__(
        self,
        x: Array,
        h: Array,
        c: Array,
    ) -> tuple[Array, Array]:
        """Apply the LSTM layer.

        :param x: Input tensor of shape (B, input_dim).
        :param h: Hidden state of shape (B, hidden_dim).
        :param c: Cell state of shape (B, hidden_dim).
        :returns: Tuple of (new hidden, new cell).
        """
        return self.cell(x, h, c)


class LSTMStack(eqx.Module):
    """Stacked LSTM."""

    input_dim: int
    hidden_dim: int
    num_layers: int
    layers: tuple[LSTMLayer, ...]

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        key: Array,
    ) -> None:
        """Initialize the LSTM stack.

        :param input_dim: Input feature dimension for the first layer.
        :param hidden_dim: Hidden state dimension for all layers.
        :param num_layers: Number of stacked LSTM layers.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param kernel_init: Initializer for the projection weights.
        :param key: PRNG key for parameter initialization.
        """
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        keys = jax.random.split(key, num_layers)
        layers: list[LSTMLayer] = []
        for idx in range(num_layers):
            layer_input_dim = input_dim if idx == 0 else hidden_dim
            layers.append(
                LSTMLayer(
                    layer_input_dim,
                    hidden_dim,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    kernel_init=kernel_init,
                    key=keys[idx],
                )
            )
        self.layers = tuple(layers)

    def __call__(self, inputs: Array) -> Array:
        """Run the LSTM stack over a sequence.

        :param inputs: Input tensor of shape (B, T, input_dim).
        :returns: Output tensor of shape (B, T, hidden_dim).
        """
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (B, T, input_dim)")
        if inputs.shape[2] != self.input_dim:
            raise ValueError("inputs last dimension must match input_dim")

        batch_size = inputs.shape[0]
        init_h = tuple(
            jnp.zeros((batch_size, self.hidden_dim), dtype=inputs.dtype)
            for _ in range(self.num_layers)
        )
        init_c = tuple(
            jnp.zeros((batch_size, self.hidden_dim), dtype=inputs.dtype)
            for _ in range(self.num_layers)
        )

        def _step(
            carry: tuple[tuple[Array, ...], tuple[Array, ...]],
            x_t: Array,
        ) -> tuple[tuple[tuple[Array, ...], tuple[Array, ...]], Array]:
            h_list, c_list = carry
            new_h: list[Array] = []
            new_c: list[Array] = []
            layer_input = x_t
            for layer, h_state, c_state in zip(self.layers, h_list, c_list):
                h_next, c_next = layer(layer_input, h_state, c_state)
                layer_input = h_next
                new_h.append(h_next)
                new_c.append(c_next)
            return (tuple(new_h), tuple(new_c)), layer_input

        inputs_time = jnp.swapaxes(inputs, 0, 1)
        (_final_h, _final_c), outputs = jax.lax.scan(
            _step,
            (init_h, init_c),
            inputs_time,
        )
        outputs = jnp.swapaxes(outputs, 0, 1)
        return outputs


class BiLSTMLayer(eqx.Module):
    """Bidirectional LSTM layer with a projection for the next layer."""

    forward: LSTMLayer
    backward: LSTMLayer
    proj: Linear

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        key: Array,
    ) -> None:
        """Initialize the bidirectional LSTM layer.

        :param input_dim: Input feature dimension.
        :param hidden_dim: Hidden state dimension per direction.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param kernel_init: Initializer for the projection weights.
        :param key: PRNG key for parameter initialization.
        """
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        forward_key, backward_key, proj_key = jax.random.split(key, 3)
        self.forward = LSTMLayer(
            input_dim,
            hidden_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            key=forward_key,
        )
        self.backward = LSTMLayer(
            input_dim,
            hidden_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            key=backward_key,
        )
        self.proj = Linear(
            in_features=hidden_dim * 2,
            out_features=hidden_dim,
            use_bias=True,
            bias_value=0.0,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            key=proj_key,
        )

    def __call__(self, inputs: Array) -> Array:
        """Run the bidirectional layer over a sequence.

        :param inputs: Input tensor of shape (B, T, input_dim).
        :returns: Output tensor of shape (B, T, hidden_dim).
        """
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (B, T, input_dim)")
        batch_size = inputs.shape[0]

        def _run_layer(layer: LSTMLayer, seq: Array) -> Array:
            init_h = jnp.zeros((batch_size, layer.cell.hidden_dim), dtype=seq.dtype)
            init_c = jnp.zeros((batch_size, layer.cell.hidden_dim), dtype=seq.dtype)

            def _step(carry: tuple[Array, Array], x_t: Array) -> tuple[tuple[Array, Array], Array]:
                h_state, c_state = carry
                h_next, c_next = layer.cell(x_t, h_state, c_state)
                return (h_next, c_next), h_next

            seq_time = jnp.swapaxes(seq, 0, 1)
            (_final_h, _final_c), outputs = jax.lax.scan(_step, (init_h, init_c), seq_time)
            return jnp.swapaxes(outputs, 0, 1)

        forward_out = _run_layer(self.forward, inputs)
        reversed_inputs = jnp.flip(inputs, axis=1)
        backward_out_reversed = _run_layer(self.backward, reversed_inputs)
        backward_out = jnp.flip(backward_out_reversed, axis=1)
        combined = jnp.concatenate([forward_out, backward_out], axis=-1)
        return self.proj(combined)


class BiLSTMStack(eqx.Module):
    """Stacked bidirectional LSTM with per-layer projections."""

    input_dim: int
    hidden_dim: int
    num_layers: int
    layers: tuple[BiLSTMLayer, ...]

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        key: Array,
    ) -> None:
        """Initialize the bidirectional LSTM stack.

        :param input_dim: Input feature dimension for the first layer.
        :param hidden_dim: Hidden state dimension per direction.
        :param num_layers: Number of stacked bidirectional layers.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param kernel_init: Initializer for the projection weights.
        :param key: PRNG key for parameter initialization.
        """
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        keys = jax.random.split(key, num_layers)
        layers: list[BiLSTMLayer] = []
        for idx in range(num_layers):
            layer_input_dim = input_dim if idx == 0 else hidden_dim
            layers.append(
                BiLSTMLayer(
                    layer_input_dim,
                    hidden_dim,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    kernel_init=kernel_init,
                    key=keys[idx],
                )
            )
        self.layers = tuple(layers)

    def __call__(self, inputs: Array) -> Array:
        """Run the bidirectional LSTM stack over a sequence.

        :param inputs: Input tensor of shape (B, T, input_dim).
        :returns: Output tensor of shape (B, T, hidden_dim).
        """
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (B, T, input_dim)")
        if inputs.shape[2] != self.input_dim:
            raise ValueError("inputs last dimension must match input_dim")
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs
