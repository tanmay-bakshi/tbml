from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from pydantic import BaseModel, Field

from tbml.nn import DropPath, GatedPositionalSelfAttention, Linear, RMSNorm, SelfAttention, SwiGLUFeedForward
from tbml.nn.init import Initializer, truncated_normal_init


class ConViTConfig(BaseModel):
    """Configuration for the ConViT model.

    :ivar image_size: Input image size as (height, width).
    :ivar patch_size: Patch size (square).
    :ivar in_channels: Number of input channels.
    :ivar d_model: Model width.
    :ivar n_heads: Number of attention heads.
    :ivar n_gpsa_layers: Number of GPSA transformer blocks.
    :ivar n_sa_layers: Number of SA transformer blocks.
    :ivar mlp_ratio: Expansion ratio for the SwiGLU hidden dimension.
    :ivar attn_dropout: Attention dropout probability.
    :ivar resid_dropout: Residual dropout probability.
    :ivar drop_path_rate: Stochastic depth rate at the final block.
    :ivar locality_strength: GPSA locality strength (alpha).
    :ivar init_std: Standard deviation for truncated normal initialization.
    """

    image_size: tuple[int, int] = Field(default=(224, 224))
    patch_size: int = Field(default=16)
    in_channels: int = Field(default=3)
    d_model: int = Field(default=384)
    n_heads: int = Field(default=9)
    n_gpsa_layers: int = Field(default=10)
    n_sa_layers: int = Field(default=2)
    mlp_ratio: float = Field(default=4.0)
    attn_dropout: float = Field(default=0.0)
    resid_dropout: float = Field(default=0.0)
    drop_path_rate: float = Field(default=0.0)
    locality_strength: float = Field(default=1.0)
    init_std: float = Field(default=0.02)


class PatchEmbed(eqx.Module):
    """Patchify images and embed patches with a linear projection."""

    image_size: tuple[int, int]
    patch_size: int
    in_channels: int
    embed_dim: int
    grid_size: tuple[int, int]
    num_patches: int
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    proj: Linear

    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        key: Array,
    ) -> None:
        """Initialize patch embedding parameters.

        :param image_size: Expected input size as (height, width).
        :param patch_size: Patch size (square).
        :param in_channels: Number of input channels.
        :param embed_dim: Output embedding dimension.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param kernel_init: Initializer for the patch projection.
        :param key: PRNG key for parameter initialization.
        """
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("image_size entries must be > 0")
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        grid_h = image_size[0] // patch_size
        grid_w = image_size[1] // patch_size
        patch_dim = patch_size * patch_size * in_channels

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.grid_size = (grid_h, grid_w)
        self.num_patches = grid_h * grid_w
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.proj = Linear(
            in_features=patch_dim,
            out_features=embed_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        """Patchify and embed an input batch.

        :param x: Input tensor of shape (B, H, W, C).
        :returns: Patch embeddings of shape (B, num_patches, embed_dim).
        :raises ValueError: If the input size is not divisible by the patch size.
        """
        if x.ndim != 4:
            raise ValueError("input must have shape (B, H, W, C)")
        bsz, height, width, channels = x.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("input spatial size must be divisible by patch_size")
        if channels != self.in_channels:
            raise ValueError("input channel count must match in_channels")

        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        num_patches = grid_h * grid_w
        patch = self.patch_size
        x = x.astype(self.dtype)
        x = x.reshape((bsz, grid_h, patch, grid_w, patch, channels))
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape((bsz, num_patches, patch * patch * channels))
        return self.proj(x)


class GPSABlock(eqx.Module):
    """Transformer block using gated positional self-attention."""

    norm1: RMSNorm
    attn: GatedPositionalSelfAttention
    drop_path1: DropPath
    norm2: RMSNorm
    mlp: SwiGLUFeedForward
    drop_path2: DropPath

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        grid_size: tuple[int, int],
        mlp_hidden_dim: int,
        locality_strength: float,
        *,
        attn_dropout: float,
        resid_dropout: float,
        drop_path_prob: float,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        qkv_kernel_init: Initializer,
        o_kernel_init: Initializer,
        mlp_kernel_init: Initializer,
        attn_implementation: str | None,
        key: Array,
    ) -> None:
        """Initialize a GPSA transformer block.

        :param d_model: Model width.
        :param n_heads: Number of attention heads.
        :param grid_size: Patch grid size as (height, width).
        :param mlp_hidden_dim: Hidden dimension of the SwiGLU MLP.
        :param locality_strength: GPSA locality strength (alpha).
        :param attn_dropout: Attention dropout probability.
        :param resid_dropout: Residual dropout probability.
        :param drop_path_prob: DropPath probability for this block.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param qkv_kernel_init: Initializer for Q/K/V projections.
        :param o_kernel_init: Initializer for output projection.
        :param mlp_kernel_init: Initializer for MLP projections.
        :param key: PRNG key for parameter initialization.
        """
        attn_key, mlp_key = jax.random.split(key, 2)

        self.norm1 = RMSNorm(d_model, dtype=dtype, param_dtype=param_dtype)
        self.attn = GatedPositionalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            grid_size=grid_size,
            locality_strength=locality_strength,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            dtype=dtype,
            param_dtype=param_dtype,
            qkv_kernel_init=qkv_kernel_init,
            o_kernel_init=o_kernel_init,
            attn_implementation=attn_implementation,
            key=attn_key,
        )
        self.drop_path1 = DropPath(drop_path_prob)
        self.norm2 = RMSNorm(d_model, dtype=dtype, param_dtype=param_dtype)
        self.mlp = SwiGLUFeedForward(
            d_model=d_model,
            hidden_dim=mlp_hidden_dim,
            resid_dropout=resid_dropout,
            dtype=dtype,
            param_dtype=param_dtype,
            gate_up_kernel_init=mlp_kernel_init,
            down_kernel_init=mlp_kernel_init,
            key=mlp_key,
        )
        self.drop_path2 = DropPath(drop_path_prob)

    def __call__(self, x: Array, *, train: bool, key: Array | None) -> Array:
        """Apply the GPSA transformer block.

        :param x: Input tensor of shape (B, T, d_model).
        :param train: Whether to enable dropout.
        :param key: PRNG key for dropout and DropPath.
        :returns: Output tensor of shape (B, T, d_model).
        """
        if key is None:
            attn_key = None
            mlp_key = None
            drop1_key = None
            drop2_key = None
        else:
            attn_key, mlp_key, drop1_key, drop2_key = jax.random.split(key, 4)

        attn_out = self.attn(self.norm1(x), train=train, key=attn_key)
        attn_out = self.drop_path1(attn_out, train=train, key=drop1_key)
        x = x + attn_out

        mlp_out = self.mlp(self.norm2(x), train=train, key=mlp_key)
        mlp_out = self.drop_path2(mlp_out, train=train, key=drop2_key)
        return x + mlp_out


class SABlock(eqx.Module):
    """Transformer block using standard self-attention."""

    norm1: RMSNorm
    attn: SelfAttention
    drop_path1: DropPath
    norm2: RMSNorm
    mlp: SwiGLUFeedForward
    drop_path2: DropPath

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_hidden_dim: int,
        *,
        attn_dropout: float,
        resid_dropout: float,
        drop_path_prob: float,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        qkv_kernel_init: Initializer,
        o_kernel_init: Initializer,
        mlp_kernel_init: Initializer,
        attn_implementation: str | None,
        key: Array,
    ) -> None:
        """Initialize a SA transformer block.

        :param d_model: Model width.
        :param n_heads: Number of attention heads.
        :param mlp_hidden_dim: Hidden dimension of the SwiGLU MLP.
        :param attn_dropout: Attention dropout probability.
        :param resid_dropout: Residual dropout probability.
        :param drop_path_prob: DropPath probability for this block.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param qkv_kernel_init: Initializer for Q/K/V projections.
        :param o_kernel_init: Initializer for output projection.
        :param mlp_kernel_init: Initializer for MLP projections.
        :param key: PRNG key for parameter initialization.
        """
        attn_key, mlp_key = jax.random.split(key, 2)

        self.norm1 = RMSNorm(d_model, dtype=dtype, param_dtype=param_dtype)
        self.attn = SelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            is_causal=False,
            dtype=dtype,
            param_dtype=param_dtype,
            qkv_kernel_init=qkv_kernel_init,
            o_kernel_init=o_kernel_init,
            attn_implementation=attn_implementation,
            key=attn_key,
        )
        self.drop_path1 = DropPath(drop_path_prob)
        self.norm2 = RMSNorm(d_model, dtype=dtype, param_dtype=param_dtype)
        self.mlp = SwiGLUFeedForward(
            d_model=d_model,
            hidden_dim=mlp_hidden_dim,
            resid_dropout=resid_dropout,
            dtype=dtype,
            param_dtype=param_dtype,
            gate_up_kernel_init=mlp_kernel_init,
            down_kernel_init=mlp_kernel_init,
            key=mlp_key,
        )
        self.drop_path2 = DropPath(drop_path_prob)

    def __call__(self, x: Array, *, train: bool, key: Array | None) -> Array:
        """Apply the SA transformer block.

        :param x: Input tensor of shape (B, T, d_model).
        :param train: Whether to enable dropout.
        :param key: PRNG key for dropout and DropPath.
        :returns: Output tensor of shape (B, T, d_model).
        """
        if key is None:
            attn_key = None
            mlp_key = None
            drop1_key = None
            drop2_key = None
        else:
            attn_key, mlp_key, drop1_key, drop2_key = jax.random.split(key, 4)

        attn_out = self.attn(self.norm1(x), train=train, key=attn_key)
        attn_out = self.drop_path1(attn_out, train=train, key=drop1_key)
        x = x + attn_out

        mlp_out = self.mlp(self.norm2(x), train=train, key=mlp_key)
        mlp_out = self.drop_path2(mlp_out, train=train, key=drop2_key)
        return x + mlp_out


class ConViT(eqx.Module):
    """ConViT backbone with GPSA and SA transformer stacks."""

    MUON_PARAM_EXCLUSION_PATTERNS: ClassVar[list[str]] = [
        r"^patch_embed\..*$",
        r"^.*_norm\d*\..*$",
    ]

    config: ConViTConfig = eqx.field(static=True)
    patch_embed: PatchEmbed
    pos_embed: Array
    gpsa_blocks: tuple[GPSABlock, ...]
    sa_blocks: tuple[SABlock, ...]
    final_norm: RMSNorm

    def __init__(
        self,
        config: ConViTConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        attn_implementation: str | None = None,
        key: Array,
    ) -> None:
        """Initialize the ConViT model.

        :param config: ConViT configuration.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param attn_implementation: Attention backend implementation.
        :param key: PRNG key for parameter initialization.
        """
        if config.image_size[0] <= 0 or config.image_size[1] <= 0:
            raise ValueError("image_size entries must be > 0")
        if config.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if config.in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if config.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if config.n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if config.n_gpsa_layers < 0:
            raise ValueError("n_gpsa_layers must be >= 0")
        if config.n_sa_layers < 0:
            raise ValueError("n_sa_layers must be >= 0")
        if config.mlp_ratio <= 0.0:
            raise ValueError("mlp_ratio must be > 0")
        if config.attn_dropout < 0.0 or config.attn_dropout >= 1.0:
            raise ValueError("attn_dropout must be in [0, 1)")
        if config.resid_dropout < 0.0 or config.resid_dropout >= 1.0:
            raise ValueError("resid_dropout must be in [0, 1)")
        if config.drop_path_rate < 0.0 or config.drop_path_rate >= 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if config.locality_strength <= 0.0:
            raise ValueError("locality_strength must be > 0")
        if config.init_std <= 0.0:
            raise ValueError("init_std must be > 0")

        total_layers = config.n_gpsa_layers + config.n_sa_layers
        if total_layers <= 0:
            raise ValueError("at least one transformer block is required")
        if config.image_size[0] % config.patch_size != 0 or config.image_size[1] % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        mlp_hidden_dim = int(config.d_model * config.mlp_ratio)
        if mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be > 0")

        init = truncated_normal_init(config.init_std)
        keys = jax.random.split(key, 2 + total_layers)
        patch_key = keys[0]
        pos_key = keys[1]
        block_keys = keys[2:]

        self.config = config
        self.patch_embed = PatchEmbed(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            key=patch_key,
        )
        self.pos_embed = init(
            pos_key,
            (1, self.patch_embed.num_patches, config.d_model),
            param_dtype,
        )

        drop_rates = _build_drop_rates(config.drop_path_rate, total_layers)
        gpsa_blocks: list[GPSABlock] = []
        sa_blocks: list[SABlock] = []

        for idx in range(config.n_gpsa_layers):
            gpsa_blocks.append(
                GPSABlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    grid_size=self.patch_embed.grid_size,
                    mlp_hidden_dim=mlp_hidden_dim,
                    locality_strength=config.locality_strength,
                    attn_dropout=config.attn_dropout,
                    resid_dropout=config.resid_dropout,
                    drop_path_prob=drop_rates[idx],
                    dtype=dtype,
                    param_dtype=param_dtype,
                    qkv_kernel_init=init,
                    o_kernel_init=init,
                    mlp_kernel_init=init,
                    attn_implementation=attn_implementation,
                    key=block_keys[idx],
                )
            )

        for idx in range(config.n_sa_layers):
            block_idx = config.n_gpsa_layers + idx
            sa_blocks.append(
                SABlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    attn_dropout=config.attn_dropout,
                    resid_dropout=config.resid_dropout,
                    drop_path_prob=drop_rates[block_idx],
                    dtype=dtype,
                    param_dtype=param_dtype,
                    qkv_kernel_init=init,
                    o_kernel_init=init,
                    mlp_kernel_init=init,
                    attn_implementation=attn_implementation,
                    key=block_keys[block_idx],
                )
            )

        self.gpsa_blocks = tuple(gpsa_blocks)
        self.sa_blocks = tuple(sa_blocks)
        self.final_norm = RMSNorm(config.d_model, dtype=dtype, param_dtype=param_dtype)

    def encode_patches(self, x: Array, *, train: bool, key: Array | None) -> Array:
        """Compute patch representations before the final normalization.

        :param x: Input tensor of shape (B, H, W, C).
        :param train: Whether to enable dropout and DropPath.
        :param key: PRNG key for dropout and DropPath.
        :returns: Patch representations of shape (B, num_patches, d_model).
        """
        x = self.patch_embed(x)
        total_blocks = len(self.gpsa_blocks) + len(self.sa_blocks)
        if key is None:
            block_keys: list[Array | None] = [None] * total_blocks
        else:
            block_keys = list(jax.random.split(key, total_blocks))

        for block, block_key in zip(self.gpsa_blocks, block_keys[: len(self.gpsa_blocks)]):
            x = block(x, train=train, key=block_key)

        if len(self.sa_blocks) > 0:
            x = x + self.pos_embed.astype(x.dtype)

        for block, block_key in zip(self.sa_blocks, block_keys[len(self.gpsa_blocks) :]):
            x = block(x, train=train, key=block_key)

        return x

    def __call__(self, x: Array, *, train: bool, key: Array | None) -> Array:
        """Compute patch representations.

        :param x: Input tensor of shape (B, H, W, C).
        :param train: Whether to enable dropout and DropPath.
        :param key: PRNG key for dropout and DropPath.
        :returns: Patch representations of shape (B, num_patches, d_model).
        """
        return self.final_norm(self.encode_patches(x, train=train, key=key))


def _build_drop_rates(drop_path_rate: float, total_layers: int) -> list[float]:
    """Build a linear DropPath schedule.

    :param drop_path_rate: DropPath rate at the final block.
    :param total_layers: Total number of transformer blocks.
    :returns: List of DropPath rates, one per block.
    """
    if total_layers <= 0:
        raise ValueError("total_layers must be > 0")
    if total_layers == 1:
        return [drop_path_rate]
    return [drop_path_rate * (idx / (total_layers - 1)) for idx in range(total_layers)]
