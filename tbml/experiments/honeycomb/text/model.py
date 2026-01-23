from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from pydantic import BaseModel, Field

from tbml.nn import DropPath, PoPESelfAttention, RMSNorm, RoPESelfAttention, SwiGLUFeedForward
from tbml.nn.init import Initializer, truncated_normal_init


class TextTransformerConfig(BaseModel):
    """Configuration for the text encoder.

    :ivar vocab_size: Token vocabulary size.
    :ivar max_seq_len: Maximum sequence length.
    :ivar d_model: Model width.
    :ivar n_heads: Number of attention heads.
    :ivar n_layers: Number of transformer blocks.
    :ivar mlp_ratio: Expansion ratio for the MLP hidden dimension.
    :ivar attn_dropout: Attention dropout probability.
    :ivar resid_dropout: Residual dropout probability.
    :ivar drop_path_rate: Stochastic depth rate at the final block.
    :ivar pope_base: Base for PoPE/RoPE frequency schedule.
    :ivar init_std: Standard deviation for truncated normal initialization.
    :ivar attn_type: Attention type ("pope" or "rope").
    :ivar embed_norm: Whether to apply RMSNorm to token embeddings.
    :ivar embed_norm_scale: Scale applied after embedding RMSNorm.
    :ivar predictor_n_layers: Number of predictor blocks.
    :ivar causal_attention: Whether encoder/predictor attention is causal.
    """

    vocab_size: int = Field(default=50257)
    max_seq_len: int = Field(default=256)
    d_model: int = Field(default=768)
    n_heads: int = Field(default=12)
    n_layers: int = Field(default=12)
    mlp_ratio: float = Field(default=4.0)
    attn_dropout: float = Field(default=0.0)
    resid_dropout: float = Field(default=0.0)
    drop_path_rate: float = Field(default=0.0)
    pope_base: float = Field(default=10000.0)
    init_std: float = Field(default=0.02)
    attn_type: str = Field(default="pope")
    embed_norm: bool = Field(default=False)
    embed_norm_scale: float = Field(default=1.0)
    predictor_n_layers: int = Field(default=2)
    causal_attention: bool = Field(default=True)


class TokenEmbedding(eqx.Module):
    """Token embedding lookup table.

    :ivar vocab_size: Vocabulary size.
    :ivar d_model: Embedding dimension.
    :ivar dtype: Compute dtype.
    :ivar param_dtype: Parameter dtype.
    :ivar use_norm: Whether to apply RMSNorm to embeddings.
    :ivar embed_scale: Scale applied after embedding RMSNorm.
    :ivar weight: Embedding matrix of shape (vocab_size, d_model).
    :ivar bias: Output bias of shape (vocab_size,).
    :ivar norm: RMSNorm module applied to embeddings when enabled.
    """

    vocab_size: int
    d_model: int
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    use_norm: bool
    embed_scale: float
    weight: Array
    bias: Array
    norm: RMSNorm | None

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        use_norm: bool = False,
        embed_scale: float = 1.0,
        kernel_init: Initializer = jax.nn.initializers.lecun_normal(),
        key: Array,
    ) -> None:
        """Initialize token embedding parameters.

        :param vocab_size: Vocabulary size.
        :param d_model: Embedding dimension.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param use_norm: Whether to apply RMSNorm to embeddings.
        :param embed_scale: Scale applied after embedding RMSNorm.
        :param kernel_init: Initializer for the embedding matrix.
        :param key: PRNG key for parameter initialization.
        """
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if embed_scale <= 0.0:
            raise ValueError("embed_scale must be > 0")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_norm = use_norm
        self.embed_scale = embed_scale
        self.weight = kernel_init(key, (vocab_size, d_model), param_dtype)
        self.bias = jnp.zeros((vocab_size,), dtype=param_dtype)
        if use_norm is True:
            self.norm = RMSNorm(d_model, dtype=dtype, param_dtype=param_dtype)
        else:
            self.norm = None

    def __call__(self, tokens: Array) -> Array:
        """Embed token ids.

        :param tokens: Token ids of shape (B, T).
        :returns: Embedded tokens of shape (B, T, d_model).
        """
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (B, T)")
        if tokens.dtype not in (jnp.int32, jnp.int64):
            raise ValueError("tokens must be integer dtype")
        embeddings = jnp.take(self.weight.astype(self.dtype), tokens, axis=0)
        if self.use_norm is True:
            if self.norm is None:
                raise ValueError("norm must be set when use_norm is True")
            embeddings = self.norm(embeddings) * self.embed_scale
        return embeddings

    def unembed(self, embeddings: Array) -> Array:
        """Project embeddings back to vocabulary logits.

        :param embeddings: Embedding tensor of shape (..., d_model).
        :returns: Logits of shape (..., vocab_size).
        """
        if embeddings.ndim < 2:
            raise ValueError("embeddings must have shape (..., d_model)")
        if embeddings.shape[-1] != self.d_model:
            raise ValueError("embeddings last dimension must match d_model")
        embeddings_f = embeddings.astype(jnp.float32)
        weight = self.weight.astype(jnp.float32)
        bias = self.bias.astype(jnp.float32)
        logits = jnp.matmul(embeddings_f, weight.T) + bias
        return logits


class TextTransformerBlock(eqx.Module):
    """Transformer block with RMSNorm, positional attention, and SwiGLU MLP."""

    norm1: RMSNorm
    attn: PoPESelfAttention | RoPESelfAttention
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
        pope_base: float,
        attn_type: str,
        is_causal: bool,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        qkv_kernel_init: Initializer,
        o_kernel_init: Initializer,
        mlp_kernel_init: Initializer,
        key: Array,
    ) -> None:
        """Initialize the text transformer block.

        :param d_model: Model width.
        :param n_heads: Number of attention heads.
        :param mlp_hidden_dim: Hidden dimension of the MLP.
        :param attn_dropout: Attention dropout probability.
        :param resid_dropout: Residual dropout probability.
        :param drop_path_prob: DropPath probability for this block.
        :param pope_base: Base for PoPE/RoPE frequencies.
        :param attn_type: Attention type ("pope" or "rope").
        :param is_causal: Whether to apply causal attention masking.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param qkv_kernel_init: Initializer for Q/K/V projections.
        :param o_kernel_init: Initializer for output projection.
        :param mlp_kernel_init: Initializer for MLP projections.
        :param key: PRNG key for parameter initialization.
        """
        norm1_key, norm2_key, attn_key, mlp_key = jax.random.split(key, 4)

        self.norm1 = RMSNorm(d_model, dtype=dtype, param_dtype=param_dtype)
        if attn_type == "pope":
            self.attn = PoPESelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                is_causal=is_causal,
                base=pope_base,
                dtype=dtype,
                param_dtype=param_dtype,
                qkv_kernel_init=qkv_kernel_init,
                o_kernel_init=o_kernel_init,
                key=attn_key,
            )
        elif attn_type == "rope":
            self.attn = RoPESelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                is_causal=is_causal,
                base=pope_base,
                dtype=dtype,
                param_dtype=param_dtype,
                qkv_kernel_init=qkv_kernel_init,
                o_kernel_init=o_kernel_init,
                key=attn_key,
            )
        else:
            raise ValueError("attn_type must be 'pope' or 'rope'")
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

    def __call__(
        self,
        x: Array,
        *,
        attention_mask: Array,
        train: bool,
        key: Array | None,
    ) -> Array:
        """Apply the transformer block.

        :param x: Input tensor of shape (B, T, d_model).
        :param attention_mask: Attention mask of shape (B, T).
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

        attn_out = self.attn(self.norm1(x), train=train, key=attn_key, attention_mask=attention_mask)
        attn_out = self.drop_path1(attn_out, train=train, key=drop1_key)
        x = x + attn_out

        mlp_out = self.mlp(self.norm2(x), train=train, key=mlp_key)
        mlp_out = self.drop_path2(mlp_out, train=train, key=drop2_key)
        return x + mlp_out


class PredictorBlock(eqx.Module):
    """Transformer block for the predictor network."""

    norm1: RMSNorm
    attn: PoPESelfAttention | RoPESelfAttention
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
        pope_base: float,
        attn_type: str,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        qkv_kernel_init: Initializer,
        o_kernel_init: Initializer,
        mlp_kernel_init: Initializer,
        is_causal: bool,
        key: Array,
    ) -> None:
        """Initialize the predictor transformer block.

        :param d_model: Model width.
        :param n_heads: Number of attention heads.
        :param mlp_hidden_dim: Hidden dimension of the MLP.
        :param attn_dropout: Attention dropout probability.
        :param resid_dropout: Residual dropout probability.
        :param drop_path_prob: DropPath probability for this block.
        :param pope_base: Base for PoPE/RoPE frequencies.
        :param attn_type: Attention type ("pope" or "rope").
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param qkv_kernel_init: Initializer for Q/K/V projections.
        :param o_kernel_init: Initializer for output projection.
        :param mlp_kernel_init: Initializer for MLP projections.
        :param is_causal: Whether to apply causal attention masking.
        :param key: PRNG key for parameter initialization.
        """
        norm1_key, norm2_key, attn_key, mlp_key = jax.random.split(key, 4)

        self.norm1 = RMSNorm(d_model, dtype=dtype, param_dtype=param_dtype)
        if attn_type == "pope":
            self.attn = PoPESelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                is_causal=is_causal,
                base=pope_base,
                dtype=dtype,
                param_dtype=param_dtype,
                qkv_kernel_init=qkv_kernel_init,
                o_kernel_init=o_kernel_init,
                key=attn_key,
            )
        elif attn_type == "rope":
            self.attn = RoPESelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                is_causal=is_causal,
                base=pope_base,
                dtype=dtype,
                param_dtype=param_dtype,
                qkv_kernel_init=qkv_kernel_init,
                o_kernel_init=o_kernel_init,
                key=attn_key,
            )
        else:
            raise ValueError("attn_type must be 'pope' or 'rope'")
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
        _ = norm1_key
        _ = norm2_key

    def __call__(
        self,
        x: Array,
        *,
        attention_mask: Array,
        train: bool,
        key: Array | None,
    ) -> Array:
        """Apply the predictor block.

        :param x: Input tensor of shape (B, T, d_model).
        :param attention_mask: Attention mask of shape (B, T).
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

        attn_out = self.attn(self.norm1(x), train=train, key=attn_key, attention_mask=attention_mask)
        attn_out = self.drop_path1(attn_out, train=train, key=drop1_key)
        x = x + attn_out

        mlp_out = self.mlp(self.norm2(x), train=train, key=mlp_key)
        mlp_out = self.drop_path2(mlp_out, train=train, key=drop2_key)
        return x + mlp_out


class Predictor(eqx.Module):
    """Predictor network applied to token-level representations."""

    dtype: jnp.dtype = eqx.field(static=True)
    max_seq_len: int
    mask_tokens: Array
    blocks: tuple[PredictorBlock, ...]
    final_norm: RMSNorm
    output_ffn: SwiGLUFeedForward

    def __init__(
        self,
        config: TextTransformerConfig,
        *,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        qkv_kernel_init: Initializer,
        o_kernel_init: Initializer,
        mlp_kernel_init: Initializer,
        key: Array,
    ) -> None:
        """Initialize the predictor network.

        :param config: Base model configuration.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param qkv_kernel_init: Initializer for Q/K/V projections.
        :param o_kernel_init: Initializer for output projection.
        :param mlp_kernel_init: Initializer for MLP projections.
        :param key: PRNG key for parameter initialization.
        """
        if config.predictor_n_layers <= 0:
            raise ValueError("predictor_n_layers must be > 0")
        if config.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")

        self.dtype = dtype
        self.max_seq_len = config.max_seq_len

        init = truncated_normal_init(config.init_std)
        keys = jax.random.split(key, 2 + config.predictor_n_layers)
        mask_key = keys[0]
        block_keys = keys[1 : 1 + config.predictor_n_layers]
        output_key = keys[-1]

        self.mask_tokens = init(mask_key, (config.max_seq_len, config.d_model), param_dtype)

        mlp_hidden_dim = int(config.d_model * config.mlp_ratio)
        if mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be > 0")

        drop_rates = _build_drop_rates(config.drop_path_rate, config.predictor_n_layers)
        blocks: list[PredictorBlock] = []
        for idx in range(config.predictor_n_layers):
            blocks.append(
                PredictorBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    attn_dropout=config.attn_dropout,
                    resid_dropout=config.resid_dropout,
                    drop_path_prob=drop_rates[idx],
                    pope_base=config.pope_base,
                    attn_type=config.attn_type,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    qkv_kernel_init=qkv_kernel_init,
                    o_kernel_init=o_kernel_init,
                    mlp_kernel_init=mlp_kernel_init,
                    is_causal=config.causal_attention,
                    key=block_keys[idx],
                )
            )
        self.blocks = tuple(blocks)
        self.final_norm = RMSNorm(config.d_model, dtype=dtype, param_dtype=param_dtype)
        self.output_ffn = SwiGLUFeedForward(
            d_model=config.d_model,
            hidden_dim=mlp_hidden_dim,
            resid_dropout=config.resid_dropout,
            dtype=dtype,
            param_dtype=param_dtype,
            gate_up_kernel_init=mlp_kernel_init,
            down_kernel_init=mlp_kernel_init,
            key=output_key,
        )

    def __call__(
        self,
        reps: Array,
        attention_mask: Array,
        mask_positions: Array,
        *,
        train: bool,
        key: Array | None,
    ) -> Array:
        """Apply the predictor to token representations.

        :param reps: Token representations of shape (B, T, d_model).
        :param attention_mask: Attention mask of shape (B, T).
        :param mask_positions: Boolean mask positions of shape (B, T).
        :param train: Whether to enable dropout.
        :param key: PRNG key for dropout and DropPath.
        :returns: Predictor outputs after the output head, shape (B, T, d_model).
        """
        if reps.ndim != 3:
            raise ValueError("reps must have shape (B, T, d_model)")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape (B, T)")
        if mask_positions.ndim != 2:
            raise ValueError("mask_positions must have shape (B, T)")
        if reps.shape[:2] != attention_mask.shape or reps.shape[:2] != mask_positions.shape:
            raise ValueError("inputs must agree on batch and sequence dimensions")
        if reps.shape[1] > self.max_seq_len:
            raise ValueError("sequence length exceeds predictor max_seq_len")

        seq_len = reps.shape[1]
        mask_tokens = self.mask_tokens[:seq_len].astype(reps.dtype)
        mask_tokens = jnp.broadcast_to(mask_tokens[None, :, :], reps.shape)
        x = jnp.where(mask_positions[..., None], mask_tokens, reps)

        if key is None:
            block_keys: list[Array | None] = [None] * len(self.blocks)
            head_key = None
        else:
            keys = list(jax.random.split(key, len(self.blocks) + 1))
            block_keys = keys[:-1]
            head_key = keys[-1]

        for block, block_key in zip(self.blocks, block_keys):
            x = block(x, attention_mask=attention_mask, train=train, key=block_key)
        x = self.final_norm(x)
        return self.output_ffn(x, train=train, key=head_key)


class TextTransformer(eqx.Module):
    """Text encoder with an optional predictor stack."""

    MUON_PARAM_EXCLUSION_PATTERNS: ClassVar[list[str]] = [
        r"^token_embed\..*$",
        r"^.*norm\d*\..*$",
        r"^final_norm\..*$",
        r"^predictor\.mask_tokens$",
        r"^predictor\.final_norm\..*$",
        r"^.*attn\.delta$",
    ]

    config: TextTransformerConfig = eqx.field(static=True)
    token_embed: TokenEmbedding
    blocks: tuple[TextTransformerBlock, ...]
    final_norm: RMSNorm
    output_ffn: SwiGLUFeedForward
    predictor: Predictor | None

    def __init__(
        self,
        config: TextTransformerConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        key: Array,
    ) -> None:
        """Initialize the text transformer.

        :param config: Text transformer configuration.
        :param dtype: Compute dtype.
        :param param_dtype: Parameter dtype.
        :param key: PRNG key for parameter initialization.
        """
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if config.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if config.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if config.n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if config.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if config.mlp_ratio <= 0.0:
            raise ValueError("mlp_ratio must be > 0")
        if config.attn_dropout < 0.0 or config.attn_dropout >= 1.0:
            raise ValueError("attn_dropout must be in [0, 1)")
        if config.resid_dropout < 0.0 or config.resid_dropout >= 1.0:
            raise ValueError("resid_dropout must be in [0, 1)")
        if config.drop_path_rate < 0.0 or config.drop_path_rate >= 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if config.init_std <= 0.0:
            raise ValueError("init_std must be > 0")
        if config.embed_norm_scale <= 0.0:
            raise ValueError("embed_norm_scale must be > 0")
        if config.pope_base <= 1.0:
            raise ValueError("pope_base must be > 1.0")
        if config.attn_type not in ("pope", "rope"):
            raise ValueError("attn_type must be 'pope' or 'rope'")
        if config.predictor_n_layers < 0:
            raise ValueError("predictor_n_layers must be >= 0")

        init = truncated_normal_init(config.init_std)
        keys = jax.random.split(key, 3 + config.n_layers)
        embed_key = keys[0]
        output_key = keys[1]
        block_keys = keys[2 : 2 + config.n_layers]
        predictor_key = keys[-1]

        self.config = config
        self.token_embed = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            use_norm=config.embed_norm,
            embed_scale=config.embed_norm_scale,
            kernel_init=init,
            key=embed_key,
        )

        mlp_hidden_dim = int(config.d_model * config.mlp_ratio)
        if mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be > 0")

        drop_rates = _build_drop_rates(config.drop_path_rate, config.n_layers)
        blocks: list[TextTransformerBlock] = []
        for idx in range(config.n_layers):
            blocks.append(
                TextTransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    attn_dropout=config.attn_dropout,
                    resid_dropout=config.resid_dropout,
                    drop_path_prob=drop_rates[idx],
                    pope_base=config.pope_base,
                    attn_type=config.attn_type,
                    is_causal=config.causal_attention,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    qkv_kernel_init=init,
                    o_kernel_init=init,
                    mlp_kernel_init=init,
                    key=block_keys[idx],
                )
            )

        self.blocks = tuple(blocks)
        self.final_norm = RMSNorm(config.d_model, dtype=dtype, param_dtype=param_dtype)
        self.output_ffn = SwiGLUFeedForward(
            d_model=config.d_model,
            hidden_dim=mlp_hidden_dim,
            resid_dropout=config.resid_dropout,
            dtype=dtype,
            param_dtype=param_dtype,
            gate_up_kernel_init=init,
            down_kernel_init=init,
            key=output_key,
        )
        if config.predictor_n_layers > 0:
            self.predictor = Predictor(
                config,
                dtype=dtype,
                param_dtype=param_dtype,
                qkv_kernel_init=init,
                o_kernel_init=init,
                mlp_kernel_init=init,
                key=predictor_key,
            )
        else:
            self.predictor = None

    def __call__(
        self,
        tokens: Array,
        attention_mask: Array,
        *,
        train: bool,
        key: Array | None,
    ) -> tuple[Array, Array]:
        """Compute per-token and pooled sequence embeddings.

        :param tokens: Token ids of shape (B, T).
        :param attention_mask: Attention mask of shape (B, T).
        :param train: Whether to enable dropout.
        :param key: PRNG key for dropout and DropPath.
        :returns: Tuple of (token_reps, pooled_reps).
        """
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (B, T)")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape (B, T)")
        if tokens.shape != attention_mask.shape:
            raise ValueError("tokens and attention_mask must have the same shape")
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        _reps_pre, reps_post, pooled = self.forward_with_intermediates(
            tokens,
            attention_mask,
            train=train,
            key=key,
        )
        return reps_post, pooled

    def forward_with_intermediates(
        self,
        tokens: Array,
        attention_mask: Array,
        *,
        train: bool,
        key: Array | None,
    ) -> tuple[Array, Array, Array]:
        """Compute pre-norm tokens, post-head tokens, and pooled embeddings.

        :param tokens: Token ids of shape (B, T).
        :param attention_mask: Attention mask of shape (B, T).
        :param train: Whether to enable dropout.
        :param key: PRNG key for dropout and DropPath.
        :returns: Tuple of (pre_norm_tokens, post_head_tokens, pooled).
        """
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (B, T)")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape (B, T)")
        if tokens.shape != attention_mask.shape:
            raise ValueError("tokens and attention_mask must have the same shape")
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        reps = self.token_embed(tokens)
        if key is None:
            block_keys: list[Array | None] = [None] * len(self.blocks)
            head_key = None
        else:
            keys = list(jax.random.split(key, len(self.blocks) + 1))
            block_keys = keys[:-1]
            head_key = keys[-1]

        for block, block_key in zip(self.blocks, block_keys):
            reps = block(reps, attention_mask=attention_mask, train=train, key=block_key)

        reps_pre = reps
        reps_norm = self.final_norm(reps_pre)
        reps_post = self.output_ffn(reps_norm, train=train, key=head_key)

        mask = attention_mask.astype(bool)
        positions = jnp.arange(reps_post.shape[1], dtype=jnp.int32)
        positions = jnp.broadcast_to(positions[None, :], mask.shape)
        masked_positions = jnp.where(mask, positions, jnp.full_like(positions, -1))
        last_idx = jnp.max(masked_positions, axis=1)
        last_idx = jnp.maximum(last_idx, 0)
        idx = last_idx[:, None, None]
        idx = jnp.broadcast_to(idx, (idx.shape[0], 1, reps_post.shape[-1]))
        pooled = jnp.take_along_axis(reps_post, idx, axis=1).squeeze(axis=1)
        return reps_pre, reps_post, pooled


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
