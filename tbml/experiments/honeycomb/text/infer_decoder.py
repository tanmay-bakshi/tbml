import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from tbml.experiments.honeycomb.text.inference import TextInference, _load_run_config, _resolve_checkpoint_dir


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Run decoder inference on masked spans.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum decoder steps (does not change encoder input truncation).",
    )
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "bfloat16", "float16"])
    return parser.parse_args()


def _resolve_bos_id(checkpoint_path: str, eos_id: int) -> int:
    """Resolve BOS token id from checkpoint metadata.

    :param checkpoint_path: Path to a checkpoint directory or file.
    :param eos_id: EOS token id fallback.
    :returns: BOS token id.
    """
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_path)
    config = _load_run_config(checkpoint_dir)
    data = config.get("data")
    if isinstance(data, dict) is False:
        return eos_id
    bos_id = data.get("bos_id")
    if isinstance(bos_id, int) is False:
        return eos_id
    if bos_id < 0:
        return eos_id
    return int(bos_id)


def _prepare_tokens(
    inference: TextInference,
    text: str,
    *,
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Tokenize text and build masks for encoder/predictor.

    :param inference: TextInference helper.
    :param text: Input string.
    :param max_seq_len: Sequence length to use.
    :returns: Tuple of (tokens, tokens_no_eos, attention_mask, mask_positions, valid_len).
    """
    token_ids = inference.tokenize(text)
    if len(token_ids) > max_seq_len - 1:
        token_ids = token_ids[: max_seq_len - 1]
    token_ids.append(inference.eos_id)
    length = len(token_ids)

    tokens = np.full((max_seq_len,), inference.pad_id, dtype=np.int32)
    tokens[:length] = np.asarray(token_ids, dtype=np.int32)
    mask_positions = tokens == inference.mask_id
    attention_mask = np.logical_and(tokens != inference.pad_id, tokens != inference.eos_id)
    attention_mask = np.logical_and(attention_mask, np.logical_not(mask_positions))
    tokens_no_eos = np.where(tokens == inference.eos_id, inference.pad_id, tokens)

    eos_positions = np.where(tokens == inference.eos_id)[0]
    if eos_positions.shape[0] > 0:
        valid_len = int(eos_positions[0])
    else:
        valid_len = int(max_seq_len)
    return tokens, tokens_no_eos, attention_mask, mask_positions, valid_len


def _find_spans(mask_positions: np.ndarray, valid_len: int) -> list[tuple[int, int]]:
    """Find contiguous masked spans within a sequence.

    :param mask_positions: Boolean array of shape (T,).
    :param valid_len: Number of valid positions to inspect.
    :returns: List of (start, end) spans (end exclusive).
    """
    spans: list[tuple[int, int]] = []
    idx = 0
    while idx < valid_len:
        if bool(mask_positions[idx]) is False:
            idx += 1
            continue
        start = idx
        while idx < valid_len and bool(mask_positions[idx]) is True:
            idx += 1
        spans.append((start, idx))
    return spans


def _decode_tokens(inference: TextInference, token_ids: list[int]) -> str:
    """Decode a list of token ids.

    :param inference: TextInference helper.
    :param token_ids: List of token ids.
    :returns: Decoded string.
    """
    pieces = [inference.decode_token(tok_id) for tok_id in token_ids]
    return "".join(pieces)


def _greedy_decode_span(
    inference: TextInference,
    memory: Array,
    *,
    bos_id: int,
    eos_id: int,
    top_k: int,
    max_steps: int,
) -> tuple[list[int], list[tuple[int, list[tuple[int, float]]]]]:
    """Greedily decode one span from the decoder.

    :param inference: TextInference helper.
    :param memory: Predictor outputs for the span of shape (1, L, D).
    :param bos_id: BOS token id.
    :param eos_id: EOS token id.
    :param top_k: Number of top candidates to report.
    :param max_steps: Maximum decode steps.
    :returns: Tuple of (decoded token ids, per-step top-k info).
    """
    model = inference.model
    generated: list[int] = []
    step_info: list[tuple[int, list[tuple[int, float]]]] = []
    if top_k <= 0:
        raise ValueError("top-k must be > 0")

    mem_len = int(memory.shape[1])
    if mem_len <= 0:
        return generated, step_info

    memory_mask = jnp.ones((1, mem_len), dtype=jnp.bool_)

    for step in range(max_steps):
        input_ids = [bos_id] + generated
        dec_in = jnp.asarray([input_ids], dtype=jnp.int32)
        dec_mask = jnp.ones(dec_in.shape, dtype=jnp.bool_)
        dec_embed = model.token_embed(dec_in)
        dec_out = model.decoder(dec_embed, dec_mask, memory, memory_mask, train=False, key=None)
        logits = model.token_embed.unembed(dec_out).astype(jnp.float32)
        step_logits = logits[0, -1]
        probs = jax.nn.softmax(step_logits, axis=-1)
        top_vals, top_ids = jax.lax.top_k(probs, top_k)
        top_vals_np = np.asarray(top_vals)
        top_ids_np = np.asarray(top_ids)
        top_pairs: list[tuple[int, float]] = []
        for idx in range(int(top_ids_np.shape[0])):
            tok_id = int(top_ids_np[idx])
            prob = float(top_vals_np[idx])
            top_pairs.append((tok_id, prob))
        step_info.append((step, top_pairs))
        next_id = int(top_ids_np[0])
        if next_id == eos_id:
            break
        generated.append(next_id)
    return generated, step_info


def main() -> None:
    """Run decoder inference with greedy sampling."""
    args = _parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be > 0")

    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint)
    run_config = _load_run_config(checkpoint_dir)
    model_config = run_config.get("model")
    if isinstance(model_config, dict) is False:
        raise ValueError("checkpoint is missing model configuration")
    if "predictor_n_layers" not in model_config or "decoder_n_layers" not in model_config:
        raise ValueError("checkpoint does not include predictor/decoder weights")

    inference = TextInference.from_checkpoint(args.checkpoint, dtype=args.dtype)
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        raise ValueError("max-seq-len must be > 0")
    max_decode_len = inference.max_seq_len
    if args.max_seq_len is not None:
        max_decode_len = min(int(args.max_seq_len), inference.max_seq_len)

    tokens, tokens_no_eos, attention_mask, mask_positions, valid_len = _prepare_tokens(
        inference,
        args.text,
        max_seq_len=inference.max_seq_len,
    )
    spans = _find_spans(mask_positions, valid_len)
    if len(spans) == 0:
        print("No mask tokens found; nothing to decode.")
        return

    model = inference.model
    tokens_jax = jnp.asarray(tokens_no_eos[None, :])
    attn_jax = jnp.asarray(attention_mask[None, :])
    mask_jax = jnp.asarray(mask_positions[None, :])
    predictor_attn = jnp.logical_or(attn_jax, mask_jax)

    _reps_pre, reps_post, _pooled = model.forward_with_intermediates(
        tokens_jax,
        attn_jax,
        train=False,
        key=None,
    )
    pred_reps = model.predictor(
        reps_post,
        predictor_attn,
        mask_jax,
        train=False,
        key=None,
    )

    bos_id = _resolve_bos_id(args.checkpoint, inference.eos_id)
    span_outputs: list[list[int]] = []

    for span_idx, (start, end) in enumerate(spans):
        memory = pred_reps[:, start:end, :]
        decoded_ids, step_info = _greedy_decode_span(
            inference,
            memory,
            bos_id=bos_id,
            eos_id=inference.eos_id,
            top_k=args.top_k,
            max_steps=max_decode_len,
        )
        span_outputs.append(decoded_ids)
        print(f"Span {span_idx + 1} ({start}:{end})")
        for step, top_pairs in step_info:
            chosen_id = top_pairs[0][0]
            chosen_tok = inference.decode_token(chosen_id)
            print(f"  Step {step + 1}: chosen {chosen_tok!r} (id={chosen_id})")
            for rank, (tok_id, prob) in enumerate(top_pairs, start=1):
                tok_str = inference.decode_token(tok_id)
                print(f"    {rank}. {tok_str!r} (id={tok_id}): {prob:.6f}")

    output_tokens: list[int] = []
    span_iter = iter(zip(spans, span_outputs))
    current = next(span_iter, None)
    idx = 0
    while idx < valid_len:
        if current is None:
            output_tokens.append(int(tokens[idx]))
            idx += 1
            continue
        (span_start, span_end), pred_ids = current
        if idx < span_start:
            output_tokens.append(int(tokens[idx]))
            idx += 1
            continue
        if idx == span_start:
            for tok_id in pred_ids:
                output_tokens.append(int(tok_id))
            idx = span_end
            current = next(span_iter, None)
            continue
        idx += 1

    output_tokens = [tok_id for tok_id in output_tokens if tok_id != inference.eos_id]
    final_text = _decode_tokens(inference, output_tokens)
    print("\nFinal decoded text:")
    print(final_text)


if __name__ == "__main__":
    main()
