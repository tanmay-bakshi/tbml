import json
import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from tokenizers import Tokenizer

from tbml.experiments.honeycomb.text.dataset import _build_tokenizer
from tbml.experiments.honeycomb.text.model import TextTransformer, TextTransformerConfig


def _dtype_from_name(name: str) -> jnp.dtype:
    """Map a dtype name to a JAX dtype.

    :param name: Dtype name string.
    :returns: JAX dtype.
    """
    if name == "float32":
        return jnp.float32
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float16":
        return jnp.float16
    raise ValueError(f"unsupported dtype: {name}")


def _resolve_checkpoint_dir(path: str) -> str:
    """Resolve a checkpoint directory from a user-provided path.

    :param path: Path to a checkpoint directory or a checkpoint file.
    :returns: Resolved checkpoint directory.
    """
    if os.path.isdir(path) is True:
        checkpoint_dir = path
    else:
        checkpoint_dir = os.path.dirname(path)
        if checkpoint_dir == "":
            raise ValueError("checkpoint path must be a directory or a file inside a directory")

    model_path = os.path.join(checkpoint_dir, "model.eqx")
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    if os.path.isfile(model_path) is False:
        raise FileNotFoundError(f"model.eqx not found in checkpoint directory: {checkpoint_dir}")
    if os.path.isfile(metadata_path) is False:
        raise FileNotFoundError(f"metadata.json not found in checkpoint directory: {checkpoint_dir}")
    return checkpoint_dir


def _load_run_config(checkpoint_dir: str) -> dict[str, Any]:
    """Load the training run configuration from a checkpoint.

    :param checkpoint_dir: Checkpoint directory path.
    :returns: Run configuration dictionary.
    """
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    config = metadata.get("config")
    if config is None:
        raise ValueError("metadata.json does not contain a config entry")
    if isinstance(config, dict) is False:
        raise ValueError("metadata config entry must be a dictionary")
    return config


def _load_model(
    checkpoint_dir: str,
    *,
    dtype: jnp.dtype,
    model_config: dict[str, Any],
) -> TextTransformer:
    """Load a TextTransformer model from a checkpoint.

    :param checkpoint_dir: Checkpoint directory path.
    :param dtype: Compute and parameter dtype.
    :param model_config: Model configuration dictionary.
    :returns: Deserialized TextTransformer model.
    """
    config = TextTransformerConfig(**model_config)
    model_path = os.path.join(checkpoint_dir, "model.eqx")
    candidates = [dtype, jnp.float32, jnp.bfloat16, jnp.float16]
    seen: set[jnp.dtype] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        model_key = jax.random.PRNGKey(0)
        model = TextTransformer(
            config,
            dtype=candidate,
            param_dtype=candidate,
            key=model_key,
        )
        try:
            return eqx.tree_deserialise_leaves(model_path, model)
        except RuntimeError as exc:
            last_error = exc
            if "changed dtype" not in str(exc):
                raise
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("failed to load model")


class TextInference:
    """Helper for loading a checkpoint and embedding text inputs."""

    _tokenizer: Tokenizer
    _model: TextTransformer
    _max_seq_len: int
    _eos_id: int
    _pad_id: int
    _mask_id: int

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        model: TextTransformer,
        max_seq_len: int,
        eos_id: int,
        pad_id: int,
        mask_id: int,
    ) -> None:
        """Initialize the inference helper.

        :param tokenizer: Tokenizer used for text preprocessing.
        :param model: Loaded TextTransformer model.
        :param max_seq_len: Maximum sequence length (including EOS).
        :param eos_id: EOS token id.
        :param pad_id: Padding token id.
        :param mask_id: Mask token id.
        """
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if eos_id < 0:
            raise ValueError("eos_id must be >= 0")
        if pad_id < 0:
            raise ValueError("pad_id must be >= 0")
        if mask_id < 0:
            raise ValueError("mask_id must be >= 0")

        self._tokenizer = tokenizer
        self._model = model
        self._max_seq_len = max_seq_len
        self._eos_id = eos_id
        self._pad_id = pad_id
        self._mask_id = mask_id

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        dtype: str | None = None,
    ) -> "TextInference":
        """Construct an inference helper from a checkpoint path.

        :param checkpoint_path: Path to a checkpoint directory or file.
        :param dtype: Optional dtype override ("float32", "bfloat16", "float16").
        :returns: TextInference instance.
        """
        checkpoint_dir = _resolve_checkpoint_dir(checkpoint_path)
        config = _load_run_config(checkpoint_dir)
        model_config = config.get("model")
        if model_config is None or isinstance(model_config, dict) is False:
            raise ValueError("run config missing model configuration")
        data_config = config.get("data")
        if data_config is None or isinstance(data_config, dict) is False:
            raise ValueError("run config missing data configuration")
        training_config = config.get("training")
        if training_config is None or isinstance(training_config, dict) is False:
            raise ValueError("run config missing training configuration")

        tokenizer_name = data_config.get("tokenizer")
        eos_token = data_config.get("eos_token")
        pad_token = data_config.get("pad_token")
        mask_token = data_config.get("mask_token")
        max_seq_len = data_config.get("max_seq_len")
        if isinstance(tokenizer_name, str) is False:
            raise ValueError("data config missing tokenizer name")
        if isinstance(eos_token, str) is False:
            raise ValueError("data config missing eos token")
        if isinstance(pad_token, str) is False:
            raise ValueError("data config missing pad token")
        if isinstance(mask_token, str) is False:
            raise ValueError("data config missing mask token")
        if isinstance(max_seq_len, int) is False:
            raise ValueError("data config missing max_seq_len")

        dtype_name: str
        if dtype is None:
            dtype_name_raw = training_config.get("dtype")
            if isinstance(dtype_name_raw, str) is False:
                raise ValueError("training config missing dtype string")
            dtype_name = dtype_name_raw
        else:
            dtype_name = dtype

        model_dtype = _dtype_from_name(dtype_name)
        tokenizer, eos_id, pad_id, mask_id = _build_tokenizer(
            tokenizer_name,
            eos_token=eos_token,
            pad_token=pad_token,
            mask_token=mask_token,
        )
        model = _load_model(checkpoint_dir, dtype=model_dtype, model_config=model_config)

        return cls(
            tokenizer=tokenizer,
            model=model,
            max_seq_len=max_seq_len,
            eos_id=eos_id,
            pad_id=pad_id,
            mask_id=mask_id,
        )

    def preprocess(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize and pad a batch of texts.

        :param texts: List of input strings.
        :returns: Tuple of (tokens, attention_mask) arrays.
        """
        if len(texts) == 0:
            raise ValueError("texts must be non-empty")
        tokens = np.full((len(texts), self._max_seq_len), self._pad_id, dtype=np.int32)
        attention_mask = np.zeros((len(texts), self._max_seq_len), dtype=np.bool_)
        for idx, text in enumerate(texts):
            encoding = self._tokenizer.encode(text, add_special_tokens=False)
            token_ids = list(encoding.ids)
            if len(token_ids) > self._max_seq_len - 1:
                token_ids = token_ids[: self._max_seq_len - 1]
            token_ids.append(self._eos_id)
            length = len(token_ids)
            tokens[idx, :length] = np.asarray(token_ids, dtype=np.int32)
            attention_mask[idx, :length] = True
        return tokens, attention_mask

    def tokenize(self, text: str) -> list[int]:
        """Tokenize a single string into ids (excluding EOS).

        :param text: Input text string.
        :returns: Token ids truncated to max_seq_len - 1.
        """
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        token_ids = list(encoding.ids)
        if len(token_ids) > self._max_seq_len - 1:
            token_ids = token_ids[: self._max_seq_len - 1]
        return token_ids

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id into a string.

        :param token_id: Token id to decode.
        :returns: Decoded token string.
        """
        return self._tokenizer.decode([token_id])

    def embed_tokens(self, tokens: np.ndarray, attention_mask: np.ndarray) -> Array:
        """Embed preprocessed token sequences.

        :param tokens: Token ids of shape (B, T).
        :param attention_mask: Boolean attention mask of shape (B, T).
        :returns: JAX array of pooled embeddings.
        """
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (B, T)")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape (B, T)")
        if tokens.shape != attention_mask.shape:
            raise ValueError("tokens and attention_mask must have the same shape")
        if tokens.shape[1] != self._max_seq_len:
            raise ValueError("tokens sequence length must match max_seq_len")
        if tokens.dtype != np.int32:
            tokens = tokens.astype(np.int32, copy=False)
        tokens_jax = jnp.asarray(tokens)
        mask_jax = jnp.asarray(attention_mask)
        return self._model(tokens_jax, mask_jax, train=False, key=None)

    def embed(self, texts: list[str]) -> Array:
        """Embed a batch of texts with the loaded model.

        :param texts: List of input strings.
        :returns: JAX array of pooled embeddings.
        """
        tokens_np, mask_np = self.preprocess(texts)
        return self.embed_tokens(tokens_np, mask_np)

    @property
    def max_seq_len(self) -> int:
        """Return the configured maximum sequence length.

        :returns: Maximum sequence length.
        """
        return self._max_seq_len

    @property
    def eos_id(self) -> int:
        """Return the EOS token id.

        :returns: EOS token id.
        """
        return self._eos_id

    @property
    def pad_id(self) -> int:
        """Return the padding token id.

        :returns: Padding token id.
        """
        return self._pad_id

    @property
    def mask_id(self) -> int:
        """Return the mask token id.

        :returns: Mask token id.
        """
        return self._mask_id
