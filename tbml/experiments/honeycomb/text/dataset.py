import json
import multiprocessing as mp
import os
import queue
from typing import Iterable, Iterator

import numpy as np
from tokenizers import Tokenizer


def _list_jsonl_files(folder: str) -> list[str]:
    """Collect JSONL file paths from a folder.

    :param folder: Directory containing JSONL files.
    :returns: Sorted list of JSONL file paths.
    """
    if os.path.isdir(folder) is False:
        raise FileNotFoundError(f"data folder not found: {folder}")
    paths: list[str] = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".jsonl") is False:
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path) is False:
            continue
        paths.append(path)
    if len(paths) == 0:
        raise FileNotFoundError(f"no .jsonl files found in folder: {folder}")
    return paths


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


def _worker_loop(
    worker_id: int,
    files: list[str],
    *,
    text_field: str,
    tokenizer_name: str,
    eos_token: str,
    pad_token: str,
    mask_token: str,
    max_seq_len: int,
    shuffle_buffer: int,
    seed: int,
    output_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    """Stream tokenized samples from JSONL files into a queue.

    :param worker_id: Worker index for logging or debugging.
    :param files: List of JSONL files to process.
    :param text_field: JSON field name containing the text.
    :param tokenizer_name: Hugging Face tokenizer identifier.
    :param eos_token: EOS token string.
    :param pad_token: Padding token string.
    :param mask_token: Masking token string.
    :param max_seq_len: Maximum sequence length (including EOS).
    :param shuffle_buffer: Buffer size for streaming shuffle.
    :param seed: Random seed for deterministic shuffling.
    :param output_queue: Multiprocessing queue for emitted samples.
    :param stop_event: Event used to signal worker shutdown.
    """
    tokenizer, eos_id, _pad_id, _mask_id = _build_tokenizer(
        tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
    )
    rng = np.random.default_rng(seed)
    buffer: list[list[int]] = []
    file_order = list(files)

    def _emit(item: list[int]) -> None:
        """Emit a tokenized sequence into the output queue.

        :param item: Tokenized sequence to enqueue.
        """
        while True:
            if stop_event.is_set():
                return
            try:
                output_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    while stop_event.is_set() is False:
        if len(file_order) > 1:
            rng.shuffle(file_order)
        for path in file_order:
            if stop_event.is_set():
                break
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if stop_event.is_set():
                        break
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = record.get(text_field)
                    if isinstance(text, str) is False:
                        continue
                    encoding = tokenizer.encode(text, add_special_tokens=False)
                    token_ids = list(encoding.ids)
                    if len(token_ids) > max_seq_len - 1:
                        token_ids = token_ids[: max_seq_len - 1]
                    token_ids.append(eos_id)

                    if shuffle_buffer > 0:
                        if len(buffer) < shuffle_buffer:
                            buffer.append(token_ids)
                            continue
                        idx = int(rng.integers(0, shuffle_buffer))
                        _emit(buffer[idx])
                        buffer[idx] = token_ids
                    else:
                        _emit(token_ids)
        if len(buffer) > 0:
            rng.shuffle(buffer)
            for item in buffer:
                _emit(item)
            buffer.clear()


class StreamingTextDataset:
    """Streaming JSONL text dataset with multiprocessing workers."""

    _files: list[str]
    _text_field: str
    _tokenizer_name: str
    _eos_token: str
    _pad_token: str
    _mask_token: str
    _max_seq_len: int
    _shuffle_buffer: int
    _num_workers: int
    _prefetch: int
    _seed: int
    _queues: list[mp.Queue]
    _processes: list[mp.Process]
    _stop_event: mp.Event
    _started: bool

    def __init__(
        self,
        folder: str,
        *,
        text_field: str,
        tokenizer_name: str,
        eos_token: str,
        pad_token: str,
        mask_token: str,
        max_seq_len: int,
        shuffle_buffer: int,
        num_workers: int,
        prefetch: int,
        seed: int,
    ) -> None:
        """Initialize the streaming dataset.

        :param folder: Folder containing JSONL files.
        :param text_field: JSON field name containing the text.
        :param tokenizer_name: Hugging Face tokenizer identifier.
        :param eos_token: EOS token string.
        :param pad_token: Padding token string.
        :param mask_token: Masking token string.
        :param max_seq_len: Maximum sequence length (including EOS).
        :param shuffle_buffer: Buffer size for streaming shuffle.
        :param num_workers: Number of worker processes.
        :param prefetch: Prefetch depth per worker.
        :param seed: Random seed for deterministic streaming order.
        """
        if max_seq_len <= 1:
            raise ValueError("max_seq_len must be > 1")
        if shuffle_buffer < 0:
            raise ValueError("shuffle_buffer must be >= 0")
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")
        if prefetch <= 0:
            raise ValueError("prefetch must be > 0")

        files = _list_jsonl_files(folder)
        effective_workers = min(num_workers, len(files))
        if effective_workers <= 0:
            raise ValueError("no JSONL files available for streaming")

        self._files = files
        self._text_field = text_field
        self._tokenizer_name = tokenizer_name
        self._eos_token = eos_token
        self._pad_token = pad_token
        self._mask_token = mask_token
        self._max_seq_len = max_seq_len
        self._shuffle_buffer = shuffle_buffer
        self._num_workers = effective_workers
        self._prefetch = prefetch
        self._seed = seed
        self._queues = []
        self._processes = []
        self._stop_event = mp.get_context("spawn").Event()
        self._started = False

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over tokenized sequences.

        :returns: Iterator yielding lists of token ids.
        """
        self._ensure_started()
        rng = np.random.default_rng(self._seed)
        buffer: list[list[int]] = []
        worker_idx = 0

        while True:
            queue_obj = self._queues[worker_idx]
            worker_idx = (worker_idx + 1) % self._num_workers
            item = queue_obj.get()
            if self._shuffle_buffer > 0:
                if len(buffer) < self._shuffle_buffer:
                    buffer.append(item)
                    continue
                idx = int(rng.integers(0, self._shuffle_buffer))
                output = buffer[idx]
                buffer[idx] = item
                yield output
            else:
                yield item

    def close(self) -> None:
        """Stop worker processes and release resources."""
        if self._started is False:
            return
        self._stop_event.set()
        for process in self._processes:
            process.join(timeout=1.0)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        self._processes = []
        self._queues = []
        self._started = False

    def tokenizer_info(self) -> tuple[int, int, int, int]:
        """Return tokenizer vocabulary and special token ids.

        :returns: Tuple of (vocab_size, eos_id, pad_id, mask_id).
        """
        tokenizer, eos_id, pad_id, mask_id = _build_tokenizer(
            self._tokenizer_name,
            eos_token=self._eos_token,
            pad_token=self._pad_token,
            mask_token=self._mask_token,
        )
        vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
        return int(vocab_size), int(eos_id), int(pad_id), int(mask_id)

    def _ensure_started(self) -> None:
        """Launch worker processes if they are not already running."""
        if self._started is True:
            return
        ctx = mp.get_context("spawn")
        queues = [ctx.Queue(maxsize=self._prefetch) for _ in range(self._num_workers)]
        seed_seq = np.random.SeedSequence(self._seed)
        worker_seeds = seed_seq.spawn(self._num_workers)
        file_splits = np.array_split(self._files, self._num_workers)

        processes: list[mp.Process] = []
        for idx, files in enumerate(file_splits):
            worker_files = list(map(str, files))
            if len(worker_files) == 0:
                continue
            worker_seed = int(worker_seeds[idx].generate_state(1)[0])
            process = ctx.Process(
                target=_worker_loop,
                args=(idx, worker_files),
                kwargs={
                    "text_field": self._text_field,
                    "tokenizer_name": self._tokenizer_name,
                    "eos_token": self._eos_token,
                    "pad_token": self._pad_token,
                    "mask_token": self._mask_token,
                    "max_seq_len": self._max_seq_len,
                    "shuffle_buffer": self._shuffle_buffer,
                    "seed": worker_seed,
                    "output_queue": queues[idx],
                    "stop_event": self._stop_event,
                },
                daemon=True,
            )
            processes.append(process)
            process.start()

        self._queues = queues
        self._processes = processes
        self._started = True


def iter_text_batches(
    dataset: StreamingTextDataset,
    *,
    batch_size: int,
    max_seq_len: int,
    pad_id: int,
) -> Iterable[np.ndarray]:
    """Yield padded batches of tokens.

    :param dataset: Streaming dataset instance.
    :param batch_size: Batch size per host.
    :param max_seq_len: Maximum sequence length.
    :param pad_id: Padding token id.
    :returns: Iterable of token batches.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")

    iterator = iter(dataset)
    while True:
        tokens = np.full((batch_size, max_seq_len), pad_id, dtype=np.int32)
        for idx in range(batch_size):
            sample = next(iterator)
            length = min(len(sample), max_seq_len)
            if length > 0:
                tokens[idx, :length] = np.asarray(sample[:length], dtype=np.int32)
        yield tokens
