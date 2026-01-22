import argparse

import jax
import jax.numpy as jnp
import numpy as np

from tbml.experiments.honeycomb.text.inference import TextInference


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Run encoder MLM inference on masked tokens.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "bfloat16", "float16"])
    return parser.parse_args()


def main() -> None:
    """Run MLM inference and print top-K predictions for each mask position."""
    args = _parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be > 0")

    inference = TextInference.from_checkpoint(args.checkpoint, dtype=args.dtype)
    tokens_np, attention_np = inference.preprocess([args.text])
    tokens = tokens_np[0]
    attention = attention_np[0]
    mask_positions = np.logical_and(tokens == inference.mask_id, attention)
    mask_indices = np.where(mask_positions)[0]
    if mask_indices.shape[0] == 0:
        print("No mask tokens found; nothing to predict.")
        return

    reps = inference.encode_tokens(tokens_np, attention_np)
    logits = inference.model.token_embed.unembed(reps).astype(jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)

    for idx in mask_indices:
        position = int(idx)
        probs_pos = probs[0, position]
        top_vals, top_ids = jax.lax.top_k(probs_pos, args.top_k)
        top_vals_np = np.asarray(top_vals)
        top_ids_np = np.asarray(top_ids)
        print(f"Mask position {position}:")
        for rank in range(int(top_ids_np.shape[0])):
            token_id = int(top_ids_np[rank])
            prob = float(top_vals_np[rank])
            token_text = inference.decode_token(token_id)
            print(f"  {rank + 1}. '{token_text}' (id={token_id}): {prob:.6f}")


if __name__ == "__main__":
    main()
