"""Muon optimizer with AdamW fallback for non-matrix parameters."""

from dataclasses import dataclass
from typing import Any, Iterable
import re

import equinox as eqx
import jax
import jax.numpy as jnp


Array = jax.Array
PyTree = Any


def _zeros_like_or_none(value: Any) -> Any:
    """Create a zeros-like array or preserve ``None``.

    :param value: Leaf value from a PyTree.
    :returns: Zeros-like array or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, jax.Array):
        if value.dtype in (jnp.float16, jnp.bfloat16):
            return jnp.zeros(value.shape, dtype=jnp.float32)
    return jnp.zeros_like(value)


class MuonWithAdamWFallbackState(eqx.Module):
    """State for Muon with AdamW fallback.

    :ivar muon_momentum: Momentum buffers for Muon parameters.
    :ivar adam_m: First-moment buffers for AdamW fallback.
    :ivar adam_v: Second-moment buffers for AdamW fallback.
    :ivar step: Optimization step counter.
    """

    muon_momentum: PyTree
    adam_m: PyTree
    adam_v: PyTree
    step: Array


@dataclass(frozen=True)
class MuonWithAdamWFallback:
    """Muon optimizer with AdamW fallback for non-matrix parameters.

    :ivar muon_learning_rate: Learning rate for Muon updates.
    :ivar muon_momentum: Momentum factor for Muon.
    :ivar muon_weight_decay: Weight decay for Muon updates.
    :ivar muon_nesterov: Whether to use Nesterov momentum in Muon.
    :ivar muon_n_steps: Number of Newton-Schulz steps.
    :ivar muon_ns_eps: Epsilon for Newton-Schulz normalization.
    :ivar muon_target_update_rms: Target RMS for Muon updates.
    :ivar adamw_learning_rate: Learning rate for AdamW fallback.
    :ivar adamw_betas: AdamW beta coefficients for fallback.
    :ivar adamw_eps: AdamW epsilon for fallback.
    :ivar adamw_weight_decay: AdamW weight decay for fallback.
    :ivar muon_mask: PyTree of bools indicating Muon parameters.
    :ivar qkv_mask: PyTree of bools indicating QKV special-case parameters.
    :ivar weight_decay_mask: PyTree of bools indicating parameters that use weight decay.
    """

    muon_learning_rate: float
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0
    muon_nesterov: bool = True
    muon_n_steps: int = 5
    muon_ns_eps: float = 1e-7
    muon_target_update_rms: float = 0.2

    adamw_learning_rate: float = 1e-3
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.01

    muon_mask: PyTree | None = None
    qkv_mask: PyTree | None = None
    weight_decay_mask: PyTree | None = None

    def init(self, params: PyTree) -> MuonWithAdamWFallbackState:
        """Initialize optimizer state.

        :param params: Parameter PyTree.
        :returns: Initialized optimizer state.
        """
        zeros = jax.tree_util.tree_map(_zeros_like_or_none, params)
        return MuonWithAdamWFallbackState(
            muon_momentum=zeros,
            adam_m=zeros,
            adam_v=zeros,
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def update(
        self,
        grads: PyTree,
        state: MuonWithAdamWFallbackState,
        params: PyTree,
    ) -> tuple[PyTree, MuonWithAdamWFallbackState]:
        """Compute parameter updates.

        :param grads: Gradient PyTree.
        :param state: Optimizer state.
        :param params: Parameter PyTree.
        :returns: Tuple of (updates, new_state).
        :raises ValueError: If gradient/parameter shapes are incompatible.
        """
        if self.muon_mask is None:
            raise ValueError("muon_mask must be provided for MuonWithAdamWFallback")
        if self.qkv_mask is None:
            raise ValueError("qkv_mask must be provided for MuonWithAdamWFallback")
        if self.weight_decay_mask is None:
            weight_decay_mask = jax.tree_util.tree_map(lambda _: True, params)
        else:
            weight_decay_mask = self.weight_decay_mask

        b1, b2 = self.adamw_betas
        if len(self.adamw_betas) != 2:
            raise ValueError("adamw_betas must have exactly two values.")
        step = state.step + jnp.asarray(1, dtype=jnp.int32)
        step_f = step.astype(jnp.float32)
        bias_correction1 = jnp.asarray(1.0, dtype=jnp.float32) - jnp.power(b1, step_f)
        bias_correction2 = jnp.asarray(1.0, dtype=jnp.float32) - jnp.power(b2, step_f)

        def _muon_orthogonalize(update: Array, use_qkv: bool) -> Array:
            """Orthogonalize with Newton–Schulz and apply Moonshot RMS-matching scaling.

            :param update: Raw (Nesterov) update matrix.
            :param use_qkv: Whether to apply the QKV special-casing.
            :returns: Orthogonalized update with RMS-matching scaling applied.
            :raises ValueError: If update is not 2D.
            """
            if update.ndim != 2:
                raise ValueError("Muon orthogonalization expects 2D matrices.")

            def _ortho_and_scale(mat: Array) -> Array:
                ortho = _newton_schulz_5(mat, steps=self.muon_n_steps, eps=self.muon_ns_eps)
                mult = _muon_match_rms_multiplier(mat.shape, self.muon_target_update_rms)
                return ortho * mult

            if use_qkv is True and update.shape[0] % 3 == 0:
                parts = jnp.split(update, 3, axis=0)
                out0 = _ortho_and_scale(parts[0])
                out1 = _ortho_and_scale(parts[1])
                out2 = _ortho_and_scale(parts[2])
                return jnp.concatenate([out0, out1, out2], axis=0)
            return _ortho_and_scale(update)

        def _update_one(
            param: Array | None,
            grad: Array | None,
            velocity: Array | None,
            m: Array | None,
            v: Array | None,
            use_muon: bool,
            use_qkv: bool,
            apply_weight_decay: bool,
        ) -> tuple[Array | None, Array | None, Array | None, Array | None]:
            if param is None or grad is None:
                return None, velocity, m, v
            if velocity is None or m is None or v is None:
                raise ValueError("Optimizer state missing for parameter.")

            param_f32 = param.astype(jnp.float32)
            grad_f32 = grad.astype(jnp.float32)
            velocity_f32 = velocity.astype(jnp.float32)
            m_f32 = m.astype(jnp.float32)
            v_f32 = v.astype(jnp.float32)

            if use_muon is True:
                if param_f32.ndim != 2:
                    raise ValueError("Muon parameters must be 2D matrices.")
                new_velocity = (self.muon_momentum * velocity_f32) + grad_f32
                if self.muon_nesterov is True:
                    raw_update = grad_f32 + self.muon_momentum * new_velocity
                else:
                    raw_update = new_velocity
                ortho_update = _muon_orthogonalize(raw_update, use_qkv)
                update = -self.muon_learning_rate * ortho_update
                if self.muon_weight_decay != 0.0 and apply_weight_decay is True:
                    update = update - (self.muon_learning_rate * self.muon_weight_decay) * param_f32
                return update, new_velocity, m_f32, v_f32

            new_m = b1 * m_f32 + (1.0 - b1) * grad_f32
            new_v = b2 * v_f32 + (1.0 - b2) * grad_f32 * grad_f32
            m_hat = new_m / bias_correction1
            v_hat = new_v / bias_correction2
            denom = jnp.sqrt(v_hat) + self.adamw_eps
            update = -self.adamw_learning_rate * m_hat / denom
            if self.adamw_weight_decay != 0.0 and param_f32.ndim >= 2 and apply_weight_decay is True:
                update = update - (self.adamw_learning_rate * self.adamw_weight_decay) * param_f32
            return update, velocity_f32, new_m, new_v

        mapped = jax.tree_util.tree_map(
            _update_one,
            params,
            grads,
            state.muon_momentum,
            state.adam_m,
            state.adam_v,
            self.muon_mask,
            self.qkv_mask,
            weight_decay_mask,
        )

        def _is_update_tuple(value: object) -> bool:
            if isinstance(value, tuple) is False or len(value) != 4:
                return False
            for item in value:
                if item is None:
                    continue
                if isinstance(item, (jax.Array, jax.core.Tracer)):
                    continue
                return False
            return True

        def _select(index: int) -> PyTree:
            return jax.tree_util.tree_map(
                lambda value: value[index],
                mapped,
                is_leaf=_is_update_tuple,
            )

        updates = _select(0)
        new_velocity = _select(1)
        new_m = _select(2)
        new_v = _select(3)
        new_state = MuonWithAdamWFallbackState(
            muon_momentum=new_velocity,
            adam_m=new_m,
            adam_v=new_v,
            step=step,
        )
        return updates, new_state


def _newton_schulz_5(matrix: Array, *, steps: int, eps: float) -> Array:
    """Apply 5th-order Newton-Schulz iteration for orthogonalization.

    :param matrix: Input matrix.
    :param steps: Number of iterations (> 0).
    :param eps: Epsilon for normalization.
    :returns: Orthogonalized matrix.
    :raises ValueError: If the input is not 2D or steps <= 0.
    """
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalization expects 2D matrices.")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    a = 3.4445
    b = -4.7750
    c = 2.0315

    work = matrix.astype(jnp.float32)
    norm = jnp.sqrt(jnp.sum(jnp.square(work.astype(jnp.float32))))
    inv_norm = jnp.asarray(1.0 / (norm + eps), dtype=jnp.float32)
    work = work * inv_norm

    transposed = work.shape[0] > work.shape[1]
    if transposed is True:
        work = jnp.swapaxes(work, 0, 1)

    def _body(_: int, current: Array) -> Array:
        a_mat = current @ jnp.swapaxes(current, 0, 1)
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        next_val = a * current + (b_mat @ current)
        return next_val

    work = jax.lax.fori_loop(0, steps, _body, work)

    if transposed is True:
        work = jnp.swapaxes(work, 0, 1)

    return work


def _muon_match_rms_multiplier(shape: tuple[int, int], target_rms: float) -> Array:
    """Compute Moonshot RMS-matching multiplier for a 2D matrix.

    This returns the factor ``target_rms * sqrt(max(A, B))`` for a matrix of
    shape ``(A, B)``.

    :param shape: Matrix shape ``(rows, cols)``.
    :param target_rms: Target RMS for the (pre-LR) Muon update (Moonshot uses 0.2).
    :returns: Scalar multiplier as a float32 JAX array.
    """
    rows, cols = shape
    max_dim_f = float(max(int(rows), int(cols)))
    max_dim = jnp.asarray(max_dim_f, dtype=jnp.float32)
    return jnp.asarray(target_rms, dtype=jnp.float32) * jnp.sqrt(max_dim)


def build_muon_masks(
    params: PyTree,
    flat_names: list[str],
    exclusion_patterns: Iterable[str],
) -> tuple[PyTree, PyTree]:
    """Build Muon and QKV masks from parameter names.

    :param params: Parameter PyTree.
    :param flat_names: Flat list of parameter names aligned to ``params`` leaves.
    :param exclusion_patterns: Regex patterns to exclude from Muon.
    :returns: Tuple of (muon_mask, qkv_mask).
    :raises ValueError: If the name list does not match the parameter leaves.
    """
    compiled = [re.compile(pattern) for pattern in exclusion_patterns]
    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    if len(flat_params) != len(flat_names):
        raise ValueError("flat_names must align with params leaves.")

    muon_flags: list[bool] = []
    qkv_flags: list[bool] = []
    for name, param in zip(flat_names, flat_params, strict=True):
        use_muon = False
        if isinstance(param, jax.Array) is True and param.ndim == 2:
            use_muon = True
            for pattern in compiled:
                if pattern.search(name) is not None:
                    use_muon = False
                    break
        muon_flags.append(use_muon)

        use_qkv = False
        if ".qkv.weight" in name:
            use_qkv = True
        if ".qkv.kernel" in name:
            use_qkv = True
        qkv_flags.append(use_qkv)

    muon_mask = jax.tree_util.tree_unflatten(tree_def, muon_flags)
    qkv_mask = jax.tree_util.tree_unflatten(tree_def, qkv_flags)
    return muon_mask, qkv_mask


def build_weight_decay_mask(
    params: PyTree,
    flat_names: list[str],
    exclusion_patterns: Iterable[str],
) -> PyTree:
    """Build a weight-decay mask from parameter names.

    :param params: Parameter PyTree.
    :param flat_names: Flat list of parameter names aligned to ``params`` leaves.
    :param exclusion_patterns: Regex patterns to exclude from weight decay.
    :returns: PyTree of bools indicating parameters that should use weight decay.
    :raises ValueError: If the name list does not match the parameter leaves.
    """
    compiled = [re.compile(pattern) for pattern in exclusion_patterns]
    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    if len(flat_params) != len(flat_names):
        raise ValueError("flat_names must align with params leaves.")

    flags: list[bool] = []
    for name, param in zip(flat_names, flat_params, strict=True):
        apply_decay = False
        if isinstance(param, jax.Array) is True:
            apply_decay = True
            for pattern in compiled:
                if pattern.search(name) is not None:
                    apply_decay = False
                    break
        flags.append(apply_decay)
    return jax.tree_util.tree_unflatten(tree_def, flags)
