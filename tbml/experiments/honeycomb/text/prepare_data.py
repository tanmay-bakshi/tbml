import argparse
import json
import multiprocessing as mp
import os
import queue
import threading
import time
from datetime import datetime
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq

from tbml.experiments.honeycomb.text.dataset import _build_tokenizer


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Prepare sharded token data for Honeycomb text training.")
    parser.add_argument("--data-paths", type=str, nargs="+", required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--eos-token", type=str, default="<|endoftext|>")
    parser.add_argument("--pad-token", type=str, default="<|pad|>")
    parser.add_argument("--mask-token", type=str, default="<|mask|>")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--emit-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument("--log-interval", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _list_parquet_files(paths: Sequence[str]) -> list[str]:
    """Collect parquet files from the provided paths.

    :param paths: Input file or directory paths.
    :returns: Sorted list of parquet file paths.
    """
    files: list[str] = []
    for path in paths:
        if os.path.isdir(path) is True:
            for root, _dirs, names in os.walk(path):
                for name in names:
                    if name.endswith(".parquet") is False:
                        continue
                    files.append(os.path.join(root, name))
        elif os.path.isfile(path) is True:
            if path.endswith(".parquet") is True:
                files.append(path)
        else:
            raise FileNotFoundError(f"path not found: {path}")
    files = sorted(set(files))
    if len(files) == 0:
        raise FileNotFoundError("no parquet files found in the provided paths")
    return files


def _estimate_rows(files: Sequence[str]) -> int:
    """Estimate total row count across parquet files.

    :param files: Parquet file paths.
    :returns: Total row count.
    """
    total = 0
    for path in files:
        parquet_file = pq.ParquetFile(path)
        if parquet_file.metadata is None:
            continue
        total += int(parquet_file.metadata.num_rows)
    return total


def _emit_batch(
    output_queue: mp.Queue,
    stop_event: mp.Event,
    batch: np.ndarray,
) -> None:
    """Emit a batch to the output queue, honoring backpressure.

    :param output_queue: Multiprocessing queue for batches.
    :param stop_event: Event signaling shutdown.
    :param batch: Batch array of shape (B, max_seq_len).
    """
    while stop_event.is_set() is False:
        try:
            output_queue.put(("batch", batch), timeout=0.1)
            return
        except queue.Full:
            continue


def _chunk_tokens(
    token_ids: list[int],
    *,
    chunk_len: int,
    stride: int,
) -> list[list[int]]:
    """Chunk token ids into fixed windows.

    :param token_ids: Token id sequence.
    :param chunk_len: Maximum number of tokens per chunk.
    :param stride: Stride between chunks.
    :returns: List of token chunks.
    """
    if chunk_len <= 0:
        raise ValueError("chunk_len must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if len(token_ids) == 0:
        return [[]]
    chunks: list[list[int]] = []
    start = 0
    while start < len(token_ids):
        end = start + chunk_len
        chunks.append(token_ids[start:end])
        start += stride
    return chunks


def _worker_loop(
    worker_id: int,
    task_queue: mp.Queue,
    *,
    text_field: str,
    tokenizer_name: str,
    eos_token: str,
    pad_token: str,
    mask_token: str,
    max_seq_len: int,
    stride: int,
    emit_batch_size: int,
    output_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    """Stream tokenized samples from parquet files into a queue.

    :param worker_id: Worker index.
    :param files: Assigned parquet files.
    :param text_field: Column name containing text.
    :param tokenizer_name: Hugging Face tokenizer identifier.
    :param eos_token: EOS token string.
    :param pad_token: Padding token string.
    :param mask_token: Mask token string.
    :param max_seq_len: Maximum sequence length (including EOS).
    :param stride: Stride between token chunks.
    :param emit_batch_size: Number of samples per emitted batch.
    :param output_queue: Multiprocessing queue for samples.
    :param stop_event: Event signaling shutdown.
    """
    if max_seq_len <= 1:
        raise ValueError("max_seq_len must be > 1")
    chunk_len = max_seq_len - 1
    tokenizer, eos_id, pad_id, _mask_id = _build_tokenizer(
        tokenizer_name,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
    )

    batch: list[np.ndarray] = []
    current_path: str | None = None
    parquet_file: pq.ParquetFile | None = None

    while stop_event.is_set() is False:
        try:
            task = task_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if task is None:
            break
        path, row_group = task
        if path != current_path:
            parquet_file = pq.ParquetFile(path)
            current_path = path
        if parquet_file is None:
            continue
        table = parquet_file.read_row_group(row_group, columns=[text_field])
        if table.num_columns == 0:
            output_queue.put(("rows", int(table.num_rows)))
            continue
        column = table.column(0).to_pylist()
        output_queue.put(("rows", int(table.num_rows)))
        for text in column:
            if stop_event.is_set() is True:
                break
            if isinstance(text, str) is False:
                continue
            encoding = tokenizer.encode(text, add_special_tokens=False)
            token_ids = list(encoding.ids)
            chunks = _chunk_tokens(token_ids, chunk_len=chunk_len, stride=stride)
            for chunk in chunks:
                sample = np.full((max_seq_len,), pad_id, dtype=np.int32)
                if len(chunk) > 0:
                    sample[: len(chunk)] = np.asarray(chunk, dtype=np.int32)
                sample[len(chunk)] = eos_id
                batch.append(sample)
                if len(batch) >= emit_batch_size:
                    _emit_batch(output_queue, stop_event, np.stack(batch, axis=0))
                    batch = []

    if len(batch) > 0 and stop_event.is_set() is False:
        _emit_batch(output_queue, stop_event, np.stack(batch, axis=0))
    if stop_event.is_set() is False:
        output_queue.put(None)


class _ShardWriter:
    """Accumulate samples into shard files."""

    def __init__(
        self,
        output_folder: str,
        *,
        shard_size: int,
        max_seq_len: int,
        pad_id: int,
        vocab_size: int,
    ) -> None:
        """Initialize the shard writer.

        :param output_folder: Target output folder.
        :param shard_size: Number of samples per shard.
        :param max_seq_len: Sequence length per sample.
        :param pad_id: Padding token id.
        :param vocab_size: Vocabulary size for token counts.
        """
        if shard_size <= 0:
            raise ValueError("shard_size must be > 0")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")

        self._output_folder = output_folder
        self._shard_size = shard_size
        self._max_seq_len = max_seq_len
        self._pad_id = pad_id
        self._vocab_size = vocab_size

        self._current: list[np.ndarray] = []
        self._current_count = 0
        self._shard_index = 0
        self.samples_written = 0
        self.tokens_written = 0
        self.token_counts = np.zeros((vocab_size,), dtype=np.int64)

    def add_batch(self, batch: np.ndarray) -> None:
        """Add a batch of samples to the current shard.

        :param batch: Batch array of shape (B, max_seq_len).
        """
        if batch.ndim != 2 or batch.shape[1] != self._max_seq_len:
            raise ValueError("batch must have shape (B, max_seq_len)")
        batch = batch.astype(np.int32, copy=False)
        start = 0
        total = int(batch.shape[0])
        while start < total:
            remaining = self._shard_size - self._current_count
            take = min(remaining, total - start)
            self._current.append(batch[start : start + take])
            self._current_count += take
            start += take
            if self._current_count >= self._shard_size:
                self._flush()

        mask = batch != self._pad_id
        self.tokens_written += int(np.sum(mask))
        if self._vocab_size > 0:
            flat = batch[mask]
            if flat.size > 0:
                self.token_counts += np.bincount(flat, minlength=self._vocab_size)

        self.samples_written += total

    def finalize(self) -> None:
        """Flush remaining samples to disk."""
        if self._current_count > 0:
            self._flush()

    def _flush(self) -> None:
        """Write the current shard to disk."""
        if self._current_count == 0:
            return
        shard = np.concatenate(self._current, axis=0)
        path = os.path.join(self._output_folder, f"shard_{self._shard_index:06d}.npy")
        np.save(path, shard)
        self._current = []
        self._current_count = 0
        self._shard_index += 1

    @property
    def shard_count(self) -> int:
        """Return number of written shards."""
        return self._shard_index


def _validate_output_folder(output_folder: str, *, overwrite: bool) -> None:
    """Ensure output folder is ready.

    :param output_folder: Output directory path.
    :param overwrite: Whether to overwrite existing contents.
    """
    if os.path.exists(output_folder) is True:
        if os.path.isdir(output_folder) is False:
            raise ValueError("output-folder must be a directory")
        contents = [name for name in os.listdir(output_folder) if name != ".DS_Store"]
        if len(contents) > 0 and overwrite is False:
            raise FileExistsError("output-folder is not empty (use --overwrite to proceed)")
    else:
        os.makedirs(output_folder, exist_ok=True)


def _build_tasks(files: list[str]) -> list[tuple[str, int]]:
    """Build row-group tasks across parquet files.

    :param files: Parquet file paths.
    :returns: List of (path, row_group) tasks.
    """
    tasks: list[tuple[str, int]] = []
    for path in files:
        parquet_file = pq.ParquetFile(path)
        num_row_groups = parquet_file.num_row_groups
        for row_group in range(num_row_groups):
            tasks.append((path, row_group))
    if len(tasks) == 0:
        raise ValueError("no row groups found in parquet files")
    return tasks


def main() -> None:
    """Entry point for data preparation."""
    args = _parse_args()
    if args.max_seq_len <= 1:
        raise ValueError("max-seq-len must be > 1")
    if args.shard_size <= 0:
        raise ValueError("shard-size must be > 0")
    if args.emit_batch_size <= 0:
        raise ValueError("emit-batch-size must be > 0")
    if args.num_workers <= 0:
        raise ValueError("num-workers must be > 0")
    if args.queue_size <= 0:
        raise ValueError("queue-size must be > 0")
    if args.log_interval <= 0.0:
        raise ValueError("log-interval must be > 0")

    files = _list_parquet_files(args.data_paths)
    total_rows = _estimate_rows(files)
    print(f"total_rows={total_rows}", flush=True)
    if args.stride == 0:
        stride = args.max_seq_len - 1
    else:
        stride = args.stride
    if stride <= 0:
        raise ValueError("stride must be > 0")

    _validate_output_folder(args.output_folder, overwrite=args.overwrite)

    tokenizer, eos_id, pad_id, mask_id = _build_tokenizer(
        args.tokenizer,
        eos_token=args.eos_token,
        pad_token=args.pad_token,
        mask_token=args.mask_token,
    )
    vocab_size = int(tokenizer.get_vocab_size())
    writer = _ShardWriter(
        args.output_folder,
        shard_size=args.shard_size,
        max_seq_len=args.max_seq_len,
        pad_id=pad_id,
        vocab_size=vocab_size,
    )

    ctx = mp.get_context("spawn")
    output_queue: mp.Queue = ctx.Queue(maxsize=args.queue_size)
    task_queue: mp.Queue = ctx.Queue(maxsize=args.queue_size)
    stop_event: mp.Event = ctx.Event()
    tasks = _build_tasks(files)

    workers: list[mp.Process] = []
    for worker_id in range(args.num_workers):
        proc = ctx.Process(
            target=_worker_loop,
            args=(worker_id, task_queue),
            kwargs={
                "text_field": args.text_field,
                "tokenizer_name": args.tokenizer,
                "eos_token": args.eos_token,
                "pad_token": args.pad_token,
                "mask_token": args.mask_token,
                "max_seq_len": args.max_seq_len,
                "stride": stride,
                "emit_batch_size": args.emit_batch_size,
                "output_queue": output_queue,
                "stop_event": stop_event,
            },
        )
        proc.start()
        workers.append(proc)

    num_workers = len(workers)

    def _feed_tasks() -> None:
        """Feed row-group tasks into the queue."""
        for task in tasks:
            if stop_event.is_set() is True:
                break
            while stop_event.is_set() is False:
                try:
                    task_queue.put(task, timeout=0.1)
                    break
                except queue.Full:
                    continue
        for _ in range(num_workers):
            while stop_event.is_set() is False:
                try:
                    task_queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue

    feeder = threading.Thread(target=_feed_tasks, daemon=True)
    feeder.start()
    completed = 0
    rows_processed = 0
    last_log = time.time()
    last_samples = 0
    last_tokens = 0

    try:
        while completed < num_workers:
            try:
                item = output_queue.get(timeout=0.1)
                got_item = True
            except queue.Empty:
                got_item = False

            now = time.time()
            if now - last_log >= args.log_interval:
                rate_samples = (writer.samples_written - last_samples) / (now - last_log)
                rate_tokens = (writer.tokens_written - last_tokens) / (now - last_log)
                print(
                    (
                        f"rows={rows_processed} "
                        f"samples={writer.samples_written} "
                        f"tokens={writer.tokens_written} "
                        f"shards={writer.shard_count} "
                        f"rate_samples={rate_samples:.1f}/s "
                        f"rate_tokens={rate_tokens:.1f}/s"
                    ),
                    flush=True,
                )
                last_log = now
                last_samples = writer.samples_written
                last_tokens = writer.tokens_written

            if got_item is False:
                continue

            if item is None:
                completed += 1
                continue

            if isinstance(item, tuple) is False:
                continue

            kind, payload = item
            if kind == "rows":
                rows_processed += int(payload)
                continue

            if kind != "batch":
                continue
            writer.add_batch(payload)

    finally:
        stop_event.set()
        for proc in workers:
            proc.join(timeout=1.0)
        feeder.join(timeout=1.0)

    writer.finalize()

    token_counts_path = os.path.join(args.output_folder, "token_counts.npy")
    np.save(token_counts_path, writer.token_counts)

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input_files": files,
        "text_field": args.text_field,
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "eos_token": args.eos_token,
        "pad_token": args.pad_token,
        "mask_token": args.mask_token,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "mask_id": mask_id,
        "max_seq_len": args.max_seq_len,
        "stride": stride,
        "shard_size": args.shard_size,
        "emit_batch_size": args.emit_batch_size,
        "num_workers": args.num_workers,
        "queue_size": args.queue_size,
        "total_rows": total_rows,
        "rows_processed": rows_processed,
        "samples_written": writer.samples_written,
        "tokens_written": writer.tokens_written,
        "shards_written": writer.shard_count,
        "dtype": "int32",
    }
    meta_path = os.path.join(args.output_folder, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        (
            f"done: samples={writer.samples_written} "
            f"tokens={writer.tokens_written} "
            f"shards={writer.shard_count}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
