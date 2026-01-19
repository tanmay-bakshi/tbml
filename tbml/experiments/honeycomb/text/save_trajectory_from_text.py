import argparse
import numpy as np

from tbml.experiments.honeycomb.text.inference import TextInference


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Save a token trajectory from a base checkpoint.")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def _last_non_eos_index(tokens: np.ndarray, mask: np.ndarray, eos_id: int) -> int:
    """Return the index of the last non-EOS token.

    :param tokens: Token ids of shape (1, T).
    :param mask: Attention mask of shape (1, T).
    :param eos_id: EOS token id.
    :returns: Index of the last non-EOS token.
    """
    length = int(np.sum(mask[0]))
    if length <= 0:
        raise ValueError("no tokens in text")
    if int(tokens[0, length - 1]) == eos_id:
        length -= 1
    if length <= 0:
        raise ValueError("text has no non-EOS tokens")
    return length - 1


def main() -> None:
    """Save the base model trajectory for a text string."""
    args = _parse_args()
    base = TextInference.from_checkpoint(args.base_checkpoint)
    tokens, attention_mask = base.preprocess([args.text])
    reps = base.encode_tokens(tokens, attention_mask)
    last_idx = _last_non_eos_index(tokens, attention_mask, base.eos_id)
    trajectory = np.asarray(reps)[0, : last_idx + 1]
    np.save(args.output, trajectory)
    print("Saved trajectory:", trajectory.shape)


if __name__ == "__main__":
    main()
