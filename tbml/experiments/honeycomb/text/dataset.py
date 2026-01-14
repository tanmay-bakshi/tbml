import json
import os
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer


def _build_tokenizer(
    tokenizer_name: str,
    *,
    eos_token: str,
    pad_token: str,
    mask_token: str,
) -> tuple[Tokenizer, int, int, int]:
    """Build a Hugging Face tokenizer and resolve special token ids.

    :param tokenizer_name: Hugging Face tokenizer identifier.
    :param eos_token: EOS token string.
    :param pad_token: Padding token string.
    :param mask_token: Masking token string.
    :returns: Tuple of (tokenizer, eos_id, pad_id, mask_id).
    """
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    if tokenizer is None:
        raise ValueError(f"tokenizer not found: {tokenizer_name}")

    tokens_to_add: list[str] = []
    if tokenizer.token_to_id(pad_token) is None:
        tokens_to_add.append(pad_token)
    if tokenizer.token_to_id(mask_token) is None:
        tokens_to_add.append(mask_token)
    if len(tokens_to_add) > 0:
        tokenizer.add_special_tokens(tokens_to_add)

    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise ValueError(f"eos token not found in tokenizer vocab: {eos_token}")
    pad_id = tokenizer.token_to_id(pad_token)
    if pad_id is None:
        raise ValueError(f"pad token not found in tokenizer vocab: {pad_token}")
    mask_id = tokenizer.token_to_id(mask_token)
    if mask_id is None:
        raise ValueError(f"mask token not found in tokenizer vocab: {mask_token}")

    return tokenizer, int(eos_id), int(pad_id), int(mask_id)


def _list_npy_shards(folder: str) -> list[str]:
    """Collect shard .npy files from a folder.

    :param folder: Directory containing shard files.
    :returns: Sorted list of shard file paths.
    """
    if os.path.isdir(folder) is False:
        raise FileNotFoundError(f"data folder not found: {folder}")
    paths: list[str] = []
    for name in sorted(os.listdir(folder)):
        if name.startswith("shard_") is False or name.endswith(".npy") is False:
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path) is False:
            continue
        paths.append(path)
    if len(paths) == 0:
        raise FileNotFoundError(f"no shard_*.npy files found in folder: {folder}")
    return paths


def _load_metadata(folder: str) -> dict[str, object]:
    """Load dataset metadata from disk.

    :param folder: Dataset folder path.
    :returns: Parsed metadata dictionary.
    """
    path = os.path.join(folder, "metadata.json")
    if os.path.isfile(path) is False:
        raise FileNotFoundError("metadata.json not found in dataset folder")
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if isinstance(metadata, dict) is False:
        raise ValueError("metadata.json must contain a JSON object")
    return metadata


class MMapTokenDataset:
    """Random-access dataset backed by mmap'ed NumPy shards."""

    _folder: str
    _metadata: dict[str, object]
    _shards: list[np.ndarray]
    _sizes: np.ndarray
    _offsets: np.ndarray
    _length: int
    _max_seq_len: int

    def __init__(
        self,
        folder: str,
    ) -> None:
        """Initialize the mmap-backed dataset.

        :param folder: Dataset folder containing shard_*.npy and metadata.json.
        """
        self._folder = folder
        self._metadata = _load_metadata(folder)
        shard_paths = _list_npy_shards(folder)
        shards: list[np.ndarray] = []
        sizes: list[int] = []
        max_seq_len = int(self._metadata.get("max_seq_len", 0))
        if max_seq_len <= 0:
            raise ValueError("metadata max_seq_len must be > 0")

        for path in shard_paths:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2:
                raise ValueError(f"shard {path} must be 2D, got shape {array.shape}")
            if int(array.shape[1]) != max_seq_len:
                raise ValueError("shard sequence length does not match metadata max_seq_len")
            shards.append(array)
            sizes.append(int(array.shape[0]))

        if len(shards) == 0:
            raise ValueError("no shard arrays loaded")
        total = int(sum(sizes))
        if total <= 0:
            raise ValueError("dataset contains no samples")

        self._shards = shards
        self._sizes = np.asarray(sizes, dtype=np.int64)
        self._offsets = np.cumsum(self._sizes)
        self._length = total
        self._max_seq_len = max_seq_len

    def __len__(self) -> int:
        """Return the number of samples."""
        return self._length

    def __getitem__(self, idx: int) -> np.ndarray:
        """Return a token sequence by global index.

        :param idx: Sample index.
        :returns: Token id array of shape (max_seq_len,).
        """
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError("index out of range")
        shard_idx = int(np.searchsorted(self._offsets, idx, side="right"))
        prev = 0 if shard_idx == 0 else int(self._offsets[shard_idx - 1])
        local_idx = int(idx - prev)
        return self._shards[shard_idx][local_idx]

    def metadata(self) -> dict[str, object]:
        """Return dataset metadata.

        :returns: Metadata dictionary.
        """
        return dict(self._metadata)

    def close(self) -> None:
        """Release dataset resources."""
        return

    def tokenizer_info(self) -> tuple[int, int, int, int]:
        """Return vocabulary size and special token ids from metadata.

        :returns: Tuple of (vocab_size, eos_id, pad_id, mask_id).
        """
        vocab_size = int(self._metadata.get("vocab_size", 0))
        eos_id = int(self._metadata.get("eos_id", -1))
        pad_id = int(self._metadata.get("pad_id", -1))
        mask_id = int(self._metadata.get("mask_id", -1))
        if vocab_size <= 0 or eos_id < 0 or pad_id < 0 or mask_id < 0:
            raise ValueError("metadata missing vocab_size/eos_id/pad_id/mask_id")
        return vocab_size, eos_id, pad_id, mask_id

    @property
    def max_seq_len(self) -> int:
        """Return the sequence length for each sample."""
        return self._max_seq_len


def iter_text_batches(
    dataset: MMapTokenDataset,
    *,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
) -> Iterable[np.ndarray]:
    """Yield shuffled batches of tokens.

    :param dataset: Dataset instance.
    :param batch_size: Batch size per host.
    :param shuffle_buffer: Buffer size for streaming-style shuffling.
    :param seed: Random seed for deterministic order.
    :returns: Iterable of token batches.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if shuffle_buffer < 0:
        raise ValueError("shuffle_buffer must be >= 0")

    total = len(dataset)
    rng = np.random.default_rng(seed)

    buffer: list[int] = []
    indices: list[int] = []

    for idx in range(total):
        if shuffle_buffer == 0:
            indices.append(idx)
        else:
            if len(buffer) < shuffle_buffer:
                buffer.append(idx)
                continue
            pick = int(rng.integers(0, shuffle_buffer))
            indices.append(buffer[pick])
            buffer[pick] = idx
        if len(indices) >= batch_size:
            batch = [dataset[i] for i in indices[:batch_size]]
            indices = indices[batch_size:]
            yield np.stack(batch, axis=0).astype(np.int32, copy=False)

    if shuffle_buffer > 0 and len(buffer) > 0:
        rng.shuffle(buffer)
        indices.extend(buffer)

    while len(indices) >= batch_size:
        batch = [dataset[i] for i in indices[:batch_size]]
        indices = indices[batch_size:]
        yield np.stack(batch, axis=0).astype(np.int32, copy=False)
