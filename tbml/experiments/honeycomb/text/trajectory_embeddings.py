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
    parser.add_argument("text", type=str, nargs="+", help="Input text(s) to tokenize.")
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


def _plot_trajectories_2d(
    trajectories: list[np.ndarray],
    labels_list: list[list[str]],
    *,
    title: str,
    output: str,
    interactive: bool,
) -> None:
    """Plot 2D trajectories with arrows.

    :param trajectories: List of PCA point arrays, each of shape (N, 2).
    :param labels_list: Token label list for each trajectory.
    :param title: Plot title.
    :param output: Output path for saved figure.
    :param interactive: Whether to display interactively.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(trajectories) == 0:
        raise ValueError("no trajectories to plot")
    if len(trajectories) != len(labels_list):
        raise ValueError("trajectories and labels_list length mismatch")
    cmap = plt.get_cmap("tab10" if len(trajectories) <= 10 else "hsv")
    colors = [cmap(idx % cmap.N) for idx in range(len(trajectories))]
    for seq_idx, (points, labels) in enumerate(zip(trajectories, labels_list, strict=True)):
        if points.shape[0] != len(labels):
            raise ValueError("labels length must match points length")
        color = colors[seq_idx]
        ax.plot(
            points[:, 0],
            points[:, 1],
            marker="o",
            color=color,
            linewidth=1.5,
            label=f"seq {seq_idx + 1}",
        )
        for idx in range(points.shape[0] - 1):
            ax.annotate(
                "",
                xy=(points[idx + 1, 0], points[idx + 1, 1]),
                xytext=(points[idx, 0], points[idx, 1]),
                arrowprops=dict(arrowstyle="->", color=color, linewidth=1.0),
            )
        for idx, label in enumerate(labels):
            ax.text(points[idx, 0], points[idx, 1], label, fontsize=8, ha="left", va="bottom")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    if len(trajectories) > 1:
        ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    if interactive:
        plt.show()
    else:
        plt.savefig(output, dpi=200)
        print(f"Wrote trajectory plot to: {output}")


def _plot_trajectories_3d(
    trajectories: list[np.ndarray],
    labels_list: list[list[str]],
    *,
    title: str,
    output: str,
    interactive: bool,
) -> None:
    """Plot 3D trajectories with arrows.

    :param trajectories: List of PCA point arrays, each of shape (N, 3).
    :param labels_list: Token label list for each trajectory.
    :param title: Plot title.
    :param output: Output path for saved figure.
    :param interactive: Whether to display interactively.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if len(trajectories) == 0:
        raise ValueError("no trajectories to plot")
    if len(trajectories) != len(labels_list):
        raise ValueError("trajectories and labels_list length mismatch")
    cmap = plt.get_cmap("tab10" if len(trajectories) <= 10 else "hsv")
    colors = [cmap(idx % cmap.N) for idx in range(len(trajectories))]
    for seq_idx, (points, labels) in enumerate(zip(trajectories, labels_list, strict=True)):
        if points.shape[0] != len(labels):
            raise ValueError("labels length must match points length")
        color = colors[seq_idx]
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            marker="o",
            color=color,
            linewidth=1.2,
            label=f"seq {seq_idx + 1}",
        )
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
                color=color,
                linewidth=1.0,
            )
        for idx, label in enumerate(labels):
            ax.text(points[idx, 0], points[idx, 1], points[idx, 2], label, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    if len(trajectories) > 1:
        ax.legend(loc="best", fontsize=8)
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
    text_inputs = list(args.text)
    if len(text_inputs) == 0:
        raise ValueError("at least one text input is required")

    token_ids_list: list[list[int]] = []
    labels_list: list[list[str]] = []
    for text in text_inputs:
        token_ids_no_eos = infer.tokenize(text)
        if len(token_ids_no_eos) == 0:
            raise ValueError("each text must contain at least one token")
        token_ids_list.append(token_ids_no_eos)
        labels_list.append([infer.decode_token(token_id) for token_id in token_ids_no_eos])

    tokens, attention_mask = infer.preprocess(text_inputs)
    embeddings = infer.encode_tokens(tokens, attention_mask)
    embeddings_np = np.asarray(embeddings)
    trajectories: list[np.ndarray] = []
    for idx, token_ids in enumerate(token_ids_list):
        trajectories.append(embeddings_np[idx, : len(token_ids)])

    all_points = _pca(np.concatenate(trajectories, axis=0), dim=args.pca_dim)
    split_points: list[np.ndarray] = []
    offset = 0
    for token_ids in token_ids_list:
        count = len(token_ids)
        split_points.append(all_points[offset : offset + count])
        offset += count

    title = "Trajectory Embeddings"
    if args.pca_dim == 2:
        _plot_trajectories_2d(
            split_points,
            labels_list,
            title=title,
            output=args.output,
            interactive=args.interactive,
        )
    else:
        _plot_trajectories_3d(
            split_points,
            labels_list,
            title=title,
            output=args.output,
            interactive=args.interactive,
        )


if __name__ == "__main__":
    main()
