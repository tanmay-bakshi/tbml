import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from tbml.experiments.honeycomb.text.inference import TextInference


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Run decoder LSTM inference on masked text.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--mask-token-input", dest="mask_token_input", action="store_true")
    parser.add_argument("--no-mask-token-input", dest="mask_token_input", action="store_false")
    parser.set_defaults(mask_token_input=None)
    return parser.parse_args()


def _resolve_checkpoint_dir(path: str) -> str:
    """Resolve a checkpoint directory from a user path.

    :param path: Checkpoint directory or file path.
    :returns: Checkpoint directory path.
    """
    if os.path.isdir(path) is True:
        return path
    directory = os.path.dirname(path)
    if directory == "":
        raise ValueError("checkpoint must be a directory or a file inside a directory")
    return directory


def _load_mask_token_input(checkpoint_dir: str) -> bool:
    """Load the mask_token_input flag from the checkpoint config if present.

    :param checkpoint_dir: Checkpoint directory path.
    :returns: Stored mask_token_input value or False when missing.
    """
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    if os.path.isfile(metadata_path) is False:
        return False
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    config = metadata.get("config")
    if isinstance(config, dict) is False:
        return False
    views = config.get("views")
    if isinstance(views, dict) is False:
        return False
    value = views.get("mask_token_input")
    if isinstance(value, bool) is False:
        return False
    return value


def _apply_mask_token_attention(
    tokens: np.ndarray,
    attention: np.ndarray,
    *,
    mask_id: int,
    pad_id: int,
    eos_id: int,
    mask_token_input: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare encoder tokens, attention mask, and mask positions.

    :param tokens: Token ids of shape (T,).
    :param attention: Attention mask of shape (T,).
    :param mask_id: Mask token id.
    :param pad_id: Padding token id.
    :param eos_id: EOS token id.
    :param mask_token_input: Whether masked tokens remain visible to attention.
    :returns: Tuple of (tokens_no_eos, encoder_attn, mask_positions).
    """
    tokens_no_eos = tokens.copy()
    tokens_no_eos = np.where(tokens_no_eos == eos_id, pad_id, tokens_no_eos)
    attention = np.logical_and(attention, tokens_no_eos != pad_id)
    mask_positions = np.logical_and(tokens_no_eos == mask_id, attention)
    encoder_attn = attention.copy()
    if mask_token_input is False:
        encoder_attn = np.logical_and(encoder_attn, np.logical_not(mask_positions))
        if np.any(encoder_attn) is False:
            fallback_idx = int(np.argmax(attention))
            encoder_attn[fallback_idx] = True
            mask_positions[fallback_idx] = False
    return tokens_no_eos, encoder_attn, mask_positions


def main() -> None:
    """Run decoder inference and print greedy predictions."""
    args = _parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be > 0")

    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint)
    default_mask_token_input = _load_mask_token_input(checkpoint_dir)
    if args.mask_token_input is None:
        mask_token_input = default_mask_token_input
    else:
        mask_token_input = args.mask_token_input

    inference = TextInference.from_checkpoint(args.checkpoint, dtype=args.dtype)
    tokens_np, attention_np = inference.preprocess([args.text])
    tokens = tokens_np[0]
    attention = attention_np[0]

    mask_positions_raw = np.logical_and(tokens == inference.mask_id, attention)
    if np.any(mask_positions_raw) is False:
        print("No mask tokens found; nothing to decode.")
        return

    eos_positions = np.where(tokens == inference.eos_id)[0]
    if eos_positions.shape[0] > 0:
        output_len = int(eos_positions[0])
    else:
        pad_positions = np.where(np.logical_not(attention))[0]
        output_len = int(pad_positions[0]) if pad_positions.shape[0] > 0 else int(tokens.shape[0])

    tokens_no_eos, encoder_attn, mask_positions = _apply_mask_token_attention(
        tokens,
        attention,
        mask_id=inference.mask_id,
        pad_id=inference.pad_id,
        eos_id=inference.eos_id,
        mask_token_input=mask_token_input,
    )

    tokens_jax = jnp.asarray(tokens_no_eos[None, :])
    encoder_attn_jax = jnp.asarray(encoder_attn[None, :])
    _reps_pre, reps_post, _pooled = inference.model.forward_with_intermediates(
        tokens_jax,
        encoder_attn_jax,
        train=False,
        key=None,
    )
    encoder_reps = reps_post[0]

    if inference.model.predictor is None:
        raise ValueError("predictor is not enabled in this checkpoint")
    predictor_attn = np.logical_or(encoder_attn, mask_positions)
    predictor_attn_jax = jnp.asarray(predictor_attn[None, :])
    mask_positions_jax = jnp.asarray(mask_positions[None, :])
    pred_reps = inference.model.predictor(
        reps_post,
        predictor_attn_jax,
        mask_positions_jax,
        train=False,
        key=None,
    )

    if inference.model.decoder is None:
        raise ValueError("decoder is not enabled in this checkpoint")
    decoder_reps = jnp.concatenate([pred_reps, encoder_reps[None, :, :]], axis=0)
    decoder_mask = np.stack([predictor_attn, encoder_attn], axis=0)
    decoder_vectors = inference.model.decoder(
        decoder_reps,
        jnp.asarray(decoder_mask),
        train=False,
        key=None,
    )
    decoder_logits = inference.model.token_embed.unembed(decoder_vectors).astype(jnp.float32)
    probs = jax.nn.softmax(decoder_logits, axis=-1)

    if output_len <= 0:
        print("No valid tokens to decode.")
        return

    logits_pred = decoder_logits[0, :output_len]
    probs_pred = probs[0, :output_len]
    pred_ids = jnp.argmax(logits_pred, axis=-1)
    pred_ids_np = np.asarray(pred_ids, dtype=np.int32)
    decoded_text = inference._tokenizer.decode(pred_ids_np.tolist(), skip_special_tokens=False)

    print("Predicted text:")
    print(decoded_text)
    print("")
    for pos in range(int(pred_ids_np.shape[0])):
        probs_pos = probs_pred[pos]
        top_vals, top_ids = jax.lax.top_k(probs_pos, args.top_k)
        top_vals_np = np.asarray(top_vals)
        top_ids_np = np.asarray(top_ids)
        print(f"Position {pos}:")
        for rank in range(int(top_ids_np.shape[0])):
            token_id = int(top_ids_np[rank])
            prob = float(top_vals_np[rank])
            token_text = inference.decode_token(token_id)
            print(f"  {rank + 1}. '{token_text}' (id={token_id}): {prob:.6f}")


if __name__ == "__main__":
    main()
