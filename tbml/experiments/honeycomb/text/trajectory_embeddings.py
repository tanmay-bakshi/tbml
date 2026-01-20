import argparse
import matplotlib.pyplot as plt
import numpy as np

from tbml.experiments.honeycomb.text.inference import TextInference


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Visualize the embedding trajectory of masked-prefix samples."
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory or model.eqx file.")
    parser.add_argument("text", type=str, help="Input text to tokenize.")
    parser.add_argument("--pca-dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--output", type=str, default="trajectory.png")
    return parser.parse_args()


def _pca(embeddings: np.ndarray, *, dim: int) -> np.ndarray:
    """Project embeddings via PCA.

    :param embeddings: Embeddings of shape (N, D).
    :param dim: Number of PCA dimensions.
    :returns: PCA scores of shape (N, dim).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N, D)")
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, v_t = np.linalg.svd(centered, full_matrices=False)
    components = v_t[: min(dim, v_t.shape[0]), :]
    scores = centered @ components.T
    if scores.shape[1] < dim:
        pad = np.zeros((scores.shape[0], dim - scores.shape[1]), dtype=scores.dtype)
        scores = np.concatenate([scores, pad], axis=1)
    return scores


def _plot_trajectory_2d(
    points: np.ndarray,
    labels: list[str],
    *,
    title: str,
    output: str,
    interactive: bool,
) -> None:
    """Plot 2D trajectory with arrows.

    :param points: PCA points of shape (N, 2).
    :param labels: Token labels for each point.
    :param title: Plot title.
    :param output: Output path for saved figure.
    :param interactive: Whether to display interactively.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(points[:, 0], points[:, 1], marker="o", color="#277da1", linewidth=1.5)
    for idx in range(points.shape[0] - 1):
        ax.annotate(
            "",
            xy=(points[idx + 1, 0], points[idx + 1, 1]),
            xytext=(points[idx, 0], points[idx, 1]),
            arrowprops=dict(arrowstyle="->", color="#577590", linewidth=1.0),
        )
    for idx, label in enumerate(labels):
        ax.text(points[idx, 0], points[idx, 1], label, fontsize=8, ha="left", va="bottom")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    if interactive:
        plt.show()
    else:
        plt.savefig(output, dpi=200)
        print(f"Wrote trajectory plot to: {output}")


def _plot_trajectory_3d(
    points: np.ndarray,
    labels: list[str],
    *,
    title: str,
    output: str,
    interactive: bool,
) -> None:
    """Plot 3D trajectory with arrows.

    :param points: PCA points of shape (N, 3).
    :param labels: Token labels for each point.
    :param title: Plot title.
    :param output: Output path for saved figure.
    :param interactive: Whether to display interactively.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points[:, 0], points[:, 1], points[:, 2], marker="o", color="#277da1", linewidth=1.2)
    for idx in range(points.shape[0] - 1):
        start = points[idx]
        delta = points[idx + 1] - points[idx]
        ax.quiver(
            start[0],
            start[1],
            start[2],
            delta[0],
            delta[1],
            delta[2],
            arrow_length_ratio=0.15,
            color="#577590",
            linewidth=1.0,
        )
    for idx, label in enumerate(labels):
        ax.text(points[idx, 0], points[idx, 1], points[idx, 2], label, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    if interactive:
        plt.show()
    else:
        plt.savefig(output, dpi=200)
        print(f"Wrote trajectory plot to: {output}")


def main() -> None:
    """Run the trajectory visualization."""
    args = _parse_args()
    if args.pca_dim not in (2, 3):
        raise ValueError("pca-dim must be 2 or 3")
    if args.interactive is False and args.output == "":
        raise ValueError("output path required when not using --interactive")

    infer = TextInference.from_checkpoint(args.checkpoint)
    token_ids_no_eos = infer.tokenize(args.text)
    if len(token_ids_no_eos) == 0:
        raise ValueError("text must contain at least one token")
    labels = [infer.decode_token(token_id) for token_id in token_ids_no_eos]

    tokens, attention_mask = infer.preprocess([args.text])
    embeddings = infer.encode_tokens(tokens, attention_mask)
    embeddings = np.asarray(embeddings)[0, : len(token_ids_no_eos)]

    points = _pca(embeddings, dim=args.pca_dim)
    title = "Trajectory of Masked-Prefix Embeddings"
    if args.pca_dim == 2:
        _plot_trajectory_2d(
            points,
            labels,
            title=title,
            output=args.output,
            interactive=args.interactive,
        )
    else:
        _plot_trajectory_3d(
            points,
            labels,
            title=title,
            output=args.output,
            interactive=args.interactive,
        )


if __name__ == "__main__":
    main()
