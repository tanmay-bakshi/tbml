import jax
import jax.numpy as jnp
from jaxtyping import Array


def _trapz(values: Array, grid: Array, *, axis: int = -1) -> Array:
    """Apply trapezoidal integration along an axis.

    :param values: Values to integrate.
    :param grid: Grid coordinates (1D).
    :param axis: Axis to integrate over.
    :returns: Integrated values.
    """
    if grid.ndim != 1:
        raise ValueError("grid must be 1D")
    if values.shape[axis] != grid.shape[0]:
        raise ValueError("grid size must match integration axis")

    moved = jnp.moveaxis(values, axis, -1)
    dx = grid[1:] - grid[:-1]
    left = moved[..., :-1]
    right = moved[..., 1:]
    integrated = jnp.sum((left + right) * dx * 0.5, axis=-1)
    return integrated


def sigreg_loss(
    embeddings: Array,
    *,
    global_step: Array,
    num_slices: int = 256,
    seed: int = 0,
    axis_name: str | None = None,
    t_min: float = -5.0,
    t_max: float = 5.0,
    num_knots: int = 17,
) -> Array:
    """Compute the SIGReg loss using the Epps-Pulley statistic.

    :param embeddings: Embeddings of shape (B, K).
    :param global_step: Global training step used to sync random directions.
    :param num_slices: Number of random projection directions.
    :param seed: Random seed for direction sampling.
    :param axis_name: Optional pmapped axis name for cross-device aggregation.
    :param t_min: Lower bound for integration points.
    :param t_max: Upper bound for integration points.
    :param num_knots: Number of integration knots.
    :returns: Scalar SIGReg loss.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (B, K)")
    if num_slices <= 0:
        raise ValueError("num_slices must be > 0")
    if num_knots <= 1:
        raise ValueError("num_knots must be > 1")
    if t_min >= t_max:
        raise ValueError("t_min must be < t_max")

    batch_size, dim = embeddings.shape
    if batch_size <= 0:
        raise ValueError("embeddings must have a positive batch size")
    if dim <= 0:
        raise ValueError("embeddings must have a positive feature dimension")

    key = jax.random.PRNGKey(seed)
    step = jnp.asarray(global_step, dtype=jnp.int32)
    key = jax.random.fold_in(key, step)
    embeddings_f32 = embeddings.astype(jnp.float32)
    directions = jax.random.normal(key, (dim, num_slices), dtype=jnp.float32)
    directions = directions / jnp.linalg.norm(directions, axis=0, keepdims=True)

    t = jnp.linspace(t_min, t_max, num_knots, dtype=jnp.float32)
    exp_f = jnp.exp(-0.5 * jnp.square(t))

    projections = jnp.matmul(embeddings_f32, directions)
    x_t = projections[:, :, None] * t[None, None, :]
    ecf_real = jnp.cos(x_t).mean(axis=0)
    ecf_imag = jnp.sin(x_t).mean(axis=0)
    if axis_name is not None:
        ecf_real = jax.lax.pmean(ecf_real, axis_name)
        ecf_imag = jax.lax.pmean(ecf_imag, axis_name)

    err = (jnp.square(ecf_real - exp_f[None, :]) + jnp.square(ecf_imag)) * exp_f[None, :]
    ep_stat = _trapz(err, t, axis=-1)

    if axis_name is not None:
        axis_size = jax.lax.psum(jnp.asarray(1, dtype=jnp.int32), axis_name)
        total_batch = jnp.asarray(batch_size, dtype=jnp.float32) * axis_size
    else:
        total_batch = jnp.asarray(batch_size, dtype=jnp.float32)

    return jnp.mean(ep_stat * total_batch)


def sigreg_loss_views(
    embeddings: Array,
    *,
    global_step: Array,
    num_slices: int = 256,
    seed: int = 0,
    axis_name: str | None = None,
    t_min: float = -5.0,
    t_max: float = 5.0,
    num_knots: int = 17,
) -> Array:
    """Compute the SIGReg loss across multiple views.

    :param embeddings: Embeddings of shape (B, V, K).
    :param global_step: Global training step used to sync random directions.
    :param num_slices: Number of random projection directions.
    :param seed: Random seed for direction sampling.
    :param axis_name: Optional pmapped axis name for cross-device aggregation.
    :param t_min: Lower bound for integration points.
    :param t_max: Upper bound for integration points.
    :param num_knots: Number of integration knots.
    :returns: Scalar SIGReg loss averaged over views.
    """
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape (B, V, K)")
    if num_slices <= 0:
        raise ValueError("num_slices must be > 0")
    if num_knots <= 1:
        raise ValueError("num_knots must be > 1")
    if t_min >= t_max:
        raise ValueError("t_min must be < t_max")

    batch_size, num_views, dim = embeddings.shape
    if batch_size <= 0:
        raise ValueError("embeddings must have a positive batch size")
    if num_views <= 0:
        raise ValueError("embeddings must have a positive number of views")
    if dim <= 0:
        raise ValueError("embeddings must have a positive feature dimension")

    key = jax.random.PRNGKey(seed)
    step = jnp.asarray(global_step, dtype=jnp.int32)
    key = jax.random.fold_in(key, step)
    embeddings_f32 = embeddings.astype(jnp.float32)
    directions = jax.random.normal(key, (dim, num_slices), dtype=jnp.float32)
    directions = directions / jnp.linalg.norm(directions, axis=0, keepdims=True)

    t = jnp.linspace(t_min, t_max, num_knots, dtype=jnp.float32)
    exp_f = jnp.exp(-0.5 * jnp.square(t))

    projections = jnp.einsum("bvk,ks->bvs", embeddings_f32, directions)
    x_t = projections[:, :, :, None] * t[None, None, None, :]
    ecf_real = jnp.cos(x_t).mean(axis=0)
    ecf_imag = jnp.sin(x_t).mean(axis=0)
    if axis_name is not None:
        ecf_real = jax.lax.pmean(ecf_real, axis_name)
        ecf_imag = jax.lax.pmean(ecf_imag, axis_name)

    err = (jnp.square(ecf_real - exp_f[None, None, :]) + jnp.square(ecf_imag)) * exp_f[None, None, :]
    ep_stat = _trapz(err, t, axis=-1)

    if axis_name is not None:
        axis_size = jax.lax.psum(jnp.asarray(1, dtype=jnp.int32), axis_name)
        total_batch = jnp.asarray(batch_size, dtype=jnp.float32) * axis_size
    else:
        total_batch = jnp.asarray(batch_size, dtype=jnp.float32)

    return jnp.mean(jnp.mean(ep_stat * total_batch, axis=-1))


def lejepa_loss(
    embeddings: Array,
    num_global_views: int,
    *,
    sigreg_weight: float,
    global_step: Array,
    num_slices: int = 256,
    seed: int = 0,
    axis_name: str | None = None,
) -> tuple[Array, Array, Array]:
    """Compute the LeJEPA loss.

    :param embeddings: Embeddings of shape (B, V, K).
    :param num_global_views: Number of global views (V_g).
    :param sigreg_weight: Weight for the SIGReg term (lambda).
    :param global_step: Global training step used to sync random directions.
    :param num_slices: Number of random projection directions.
    :param seed: Random seed for direction sampling.
    :param axis_name: Optional pmapped axis name for cross-device aggregation.
    :returns: Tuple of (total loss, prediction loss, SIGReg loss).
    """
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape (B, V, K)")
    if num_global_views <= 0:
        raise ValueError("num_global_views must be > 0")
    if sigreg_weight < 0.0 or sigreg_weight > 1.0:
        raise ValueError("sigreg_weight must be in [0, 1]")

    batch_size, num_views, dim = embeddings.shape
    if num_global_views > num_views:
        raise ValueError("num_global_views must be <= number of views")
    if batch_size <= 0:
        raise ValueError("embeddings must have a positive batch size")
    if dim <= 0:
        raise ValueError("embeddings must have a positive feature dimension")

    global_embeddings = embeddings[:, :num_global_views, :]
    centers = jnp.mean(global_embeddings, axis=1)
    diffs = embeddings - centers[:, None, :]
    pred_loss = jnp.mean(jnp.sum(jnp.square(diffs), axis=-1))

    sigreg = sigreg_loss_views(
        embeddings,
        global_step=global_step,
        num_slices=num_slices,
        seed=seed,
        axis_name=axis_name,
    )

    total = (1.0 - sigreg_weight) * pred_loss + sigreg_weight * sigreg
    return total, pred_loss, sigreg
