import argparse
import json
import math
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from tbml.experiments.honeycomb.text.inference import PolicyInference
from tbml.experiments.honeycomb.text.train_policy import PolicyTransformer


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :returns: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Trajectory matching via policy-guided search.")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--policy-checkpoint", type=str, required=True)
    parser.add_argument("--reference-text", type=str, required=True)
    parser.add_argument("--candidate-prefix", type=str, default="")
    parser.add_argument("--top-mass", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-simulations", type=int, default=128)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def _prepare_tokens(
    token_ids: list[int],
    *,
    max_seq_len: int,
    eos_id: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare padded token arrays for the base model.

    :param token_ids: Token ids without EOS.
    :param max_seq_len: Maximum sequence length.
    :param eos_id: EOS token id.
    :param pad_id: Padding token id.
    :returns: Tuple of (tokens, attention_mask) arrays.
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")
    ids = list(token_ids)
    if len(ids) > max_seq_len - 1:
        ids = ids[: max_seq_len - 1]
    ids.append(eos_id)
    tokens = np.full((1, max_seq_len), pad_id, dtype=np.int32)
    attention_mask = np.zeros((1, max_seq_len), dtype=np.bool_)
    tokens[0, : len(ids)] = np.asarray(ids, dtype=np.int32)
    attention_mask[0, : len(ids)] = True
    return tokens, attention_mask


def _compute_reps(
    base: PolicyInference,
    token_ids: list[int],
    ref_reps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Compute base representations and trajectory score.

    :param base: Policy inference helper.
    :param token_ids: Candidate token ids without EOS.
    :param ref_reps: Reference trajectory array of shape (R, D).
    :returns: Tuple of (token representations, last representation, score).
    """
    if len(token_ids) == 0:
        return np.zeros((0, ref_reps.shape[1]), dtype=np.float32), None, 0.0
    tokens, attention_mask = _prepare_tokens(
        token_ids,
        max_seq_len=base.base.max_seq_len,
        eos_id=base.base.eos_id,
        pad_id=base.base.pad_id,
    )
    reps = base.encode_tokens(tokens, attention_mask)
    reps_np = np.asarray(reps)[0, : len(token_ids)]
    ref_len = ref_reps.shape[0]
    compare_len = min(len(token_ids), ref_len)
    if compare_len == 0:
        score = 0.0
    else:
        diff = reps_np[:compare_len] - ref_reps[:compare_len]
        score = float(np.sum(np.linalg.norm(diff, axis=1)))
    return reps_np, reps_np[-1], score


def _top_mass_candidates(
    logits: Array,
    *,
    top_mass: float,
    temperature: float,
) -> list[tuple[int, float]]:
    """Select candidate tokens by cumulative probability mass.

    :param logits: Logits array of shape (vocab_size,).
    :param top_mass: Cumulative probability mass threshold in (0, 1].
    :returns: List of (token_id, probability) pairs.
    """
    if top_mass <= 0.0 or top_mass > 1.0:
        raise ValueError("top-mass must be in (0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    scaled = logits / temperature
    probs = np.asarray(jax.nn.softmax(scaled, axis=-1))
    indices = np.argsort(-probs)
    selected: list[tuple[int, float]] = []
    cumulative = 0.0
    for idx in indices:
        prob = float(probs[int(idx)])
        cumulative += prob
        selected.append((int(idx), prob))
        if cumulative >= top_mass and len(selected) >= 1:
            break
    if len(selected) == 0:
        selected.append((int(indices[0]), float(probs[int(indices[0])])) )
    return selected


class SearchNode:
    """MCTS node for trajectory search."""

    _SMALL: ClassVar[float] = 1e-8

    tokens: tuple[int, ...]
    depth: int
    prior: float
    score: float | None
    value_sum: float
    visits: int
    expanded: bool
    last_rep: np.ndarray | None
    children: dict[int, "SearchNode"]

    def __init__(
        self,
        tokens: tuple[int, ...],
        *,
        depth: int,
        prior: float,
    ) -> None:
        """Initialize a search node.

        :param tokens: Token ids along the path to this node.
        :param depth: Depth in the search tree.
        :param prior: Prior probability from the policy.
        """
        self.tokens = tokens
        self.depth = depth
        self.prior = prior
        self.score = None
        self.value_sum = 0.0
        self.visits = 0
        self.expanded = False
        self.last_rep = None
        self.children = {}

    def q_value(self) -> float:
        """Return the mean value for this node.

        :returns: Mean value.
        """
        if self.visits == 0:
            return 0.0
        return self.value_sum / float(self.visits)


def _ensure_node_state(
    node: SearchNode,
    *,
    base: PolicyInference,
    ref_reps: np.ndarray,
) -> None:
    """Ensure the node has computed score and last representation.

    :param node: Search node.
    :param base: Policy inference helper.
    :param ref_reps: Reference trajectory array.
    """
    if node.score is not None:
        return
    reps, last_rep, score = _compute_reps(base, list(node.tokens), ref_reps)
    _ = reps
    node.last_rep = last_rep
    node.score = score


def _expand_node(
    node: SearchNode,
    *,
    base: PolicyInference,
    policy: PolicyTransformer,
    ref_reps: np.ndarray,
    top_mass: float,
    temperature: float,
) -> None:
    """Expand a node using the policy model.

    :param node: Node to expand.
    :param base: Policy inference helper.
    :param policy: Policy model.
    :param ref_reps: Reference trajectory array.
    :param top_mass: Cumulative probability threshold.
    """
    if node.expanded is True:
        return
    if node.depth >= ref_reps.shape[0]:
        node.expanded = True
        return

    _ensure_node_state(node, base=base, ref_reps=ref_reps)
    if node.depth == 0 and len(node.tokens) == 0:
        current_rep = policy.start_token.astype(policy.dtype)
    else:
        if node.last_rep is None:
            raise ValueError("node.last_rep must be populated before expansion")
        current_rep = jnp.asarray(node.last_rep, dtype=policy.dtype)

    ref_rep = jnp.asarray(ref_reps[node.depth], dtype=policy.dtype)
    pair = jnp.stack([current_rep, ref_rep], axis=0)[None, :, :]
    logits = policy(pair, train=False, key=None)
    logits_next = logits[1]
    candidates = _top_mass_candidates(logits_next, top_mass=top_mass, temperature=temperature)

    for token_id, prior in candidates:
        child_tokens = node.tokens + (token_id,)
        node.children[token_id] = SearchNode(
            child_tokens,
            depth=node.depth + 1,
            prior=prior,
        )
    node.expanded = True


def _select_child(node: SearchNode, *, c_puct: float) -> SearchNode:
    """Select a child node using a PUCT-style score.

    :param node: Parent node.
    :param c_puct: Exploration constant.
    :returns: Selected child node.
    """
    if len(node.children) == 0:
        raise ValueError("cannot select from empty children")
    best_child: SearchNode | None = None
    best_score = -1.0e9
    parent_visits = max(node.visits, 1)
    for child in node.children.values():
        q_val = child.q_value()
        u_val = c_puct * child.prior * math.sqrt(parent_visits) / (1.0 + child.visits)
        score = q_val + u_val
        if score > best_score:
            best_score = score
            best_child = child
    if best_child is None:
        raise ValueError("failed to select a child")
    return best_child


def _run_search(
    *,
    base: PolicyInference,
    policy: PolicyTransformer,
    ref_reps: np.ndarray,
    root: SearchNode,
    top_mass: float,
    temperature: float,
    c_puct: float,
    num_simulations: int,
    max_depth: int,
) -> None:
    """Run MCTS-style search from the root.

    :param base: Policy inference helper.
    :param policy: Policy model.
    :param ref_reps: Reference trajectory array.
    :param root: Root node.
    :param top_mass: Cumulative probability threshold.
    :param c_puct: Exploration constant.
    :param num_simulations: Number of simulations.
    :param max_depth: Maximum search depth (0 = use reference length).
    """
    if num_simulations <= 0:
        raise ValueError("num_simulations must be > 0")
    if c_puct <= 0.0:
        raise ValueError("c-puct must be > 0")

    depth_limit = ref_reps.shape[0]
    if max_depth > 0:
        depth_limit = min(depth_limit, max_depth)

    for _ in range(num_simulations):
        node = root
        path = [node]
        while node.expanded is True and len(node.children) > 0:
            node = _select_child(node, c_puct=c_puct)
            path.append(node)

        _ensure_node_state(node, base=base, ref_reps=ref_reps)
        if node.depth < depth_limit:
            _expand_node(
                node,
                base=base,
                policy=policy,
                ref_reps=ref_reps,
                top_mass=top_mass,
                temperature=temperature,
            )
        value = -float(node.score if node.score is not None else 0.0)
        for item in path:
            item.visits += 1
            item.value_sum += value


def _best_child(node: SearchNode) -> SearchNode | None:
    """Select the best child based on value or prior.

    :param node: Parent node.
    :returns: Best child or None if no children exist.
    """
    if len(node.children) == 0:
        return None
    visited_children = [child for child in node.children.values() if child.visits > 0]
    use_prior = len(visited_children) == 0

    best_child: SearchNode | None = None
    best_score = -1.0e9
    for child in node.children.values():
        if use_prior is True:
            score = child.prior
        else:
            score = child.q_value()
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def _build_best_line(
    root: SearchNode,
    *,
    depth_limit: int,
) -> list[int]:
    """Build the best line by greedily selecting children.

    :param root: Root search node.
    :param depth_limit: Maximum depth for the line.
    :returns: List of token ids along the best line (including prefix).
    """
    tokens = list(root.tokens)
    current = root
    while current.depth < depth_limit:
        child = _best_child(current)
        if child is None:
            break
        tokens.append(child.tokens[-1])
        current = child
    return tokens


def main() -> None:
    """Run trajectory-guided policy search."""
    args = _parse_args()
    if args.top_mass <= 0.0 or args.top_mass > 1.0:
        raise ValueError("top-mass must be in (0, 1]")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    if args.num_simulations <= 0:
        raise ValueError("num-simulations must be > 0")
    if args.c_puct <= 0.0:
        raise ValueError("c-puct must be > 0")
    if args.max_depth < 0:
        raise ValueError("max-depth must be >= 0")

    infer = PolicyInference.from_checkpoints(
        args.base_checkpoint,
        args.policy_checkpoint,
    )
    base = infer.base
    policy = infer.policy_model

    ref_tokens = base.tokenize(args.reference_text)
    if len(ref_tokens) == 0:
        raise ValueError("reference text must contain at least one token")

    ref_tokens = ref_tokens[: base.max_seq_len - 1]
    ref_tokens_arr, ref_mask = _prepare_tokens(
        ref_tokens,
        max_seq_len=base.max_seq_len,
        eos_id=base.eos_id,
        pad_id=base.pad_id,
    )
    ref_reps = base.encode_tokens(ref_tokens_arr, ref_mask)
    ref_reps_np = np.asarray(ref_reps)[0, : len(ref_tokens)]

    prefix_tokens = base.tokenize(args.candidate_prefix) if args.candidate_prefix != "" else []
    prefix_tokens = prefix_tokens[: base.max_seq_len - 1]

    root_tokens = tuple(prefix_tokens)
    root = SearchNode(root_tokens, depth=len(root_tokens), prior=1.0)
    _ensure_node_state(root, base=infer, ref_reps=ref_reps_np)

    if root.depth < ref_reps_np.shape[0]:
        _expand_node(
            root,
            base=infer,
            policy=policy,
            ref_reps=ref_reps_np,
            top_mass=args.top_mass,
            temperature=args.temperature,
        )
        _run_search(
            base=infer,
            policy=policy,
            ref_reps=ref_reps_np,
            root=root,
            top_mass=args.top_mass,
            temperature=args.temperature,
            c_puct=args.c_puct,
            num_simulations=args.num_simulations,
            max_depth=args.max_depth,
        )

    children_payload: list[dict[str, object]] = []
    for token_id, child in root.children.items():
        mean_value = child.q_value() if child.visits > 0 else None
        children_payload.append(
            {
                "token_id": token_id,
                "token_text": base.decode_token(token_id),
                "prior": child.prior,
                "visits": child.visits,
                "value_sum": child.value_sum,
                "mean_value": mean_value,
            }
        )

    children_payload.sort(key=lambda item: float(item.get("mean_value") or -1.0e9), reverse=True)
    depth_limit = ref_reps_np.shape[0]
    if args.max_depth > 0:
        depth_limit = min(depth_limit, args.max_depth)
    best_line_tokens = _build_best_line(root, depth_limit=depth_limit)
    best_line_text = "".join([base.decode_token(token_id) for token_id in best_line_tokens])

    print("Best line tokens:", [base.decode_token(token_id) for token_id in best_line_tokens])
    print("Best line text:", repr(best_line_text))

    payload = {
        "reference_length": int(ref_reps_np.shape[0]),
        "prefix_length": int(len(prefix_tokens)),
        "top_mass": args.top_mass,
        "temperature": args.temperature,
        "num_simulations": args.num_simulations,
        "c_puct": args.c_puct,
        "children": children_payload,
    }

    output = json.dumps(payload, indent=2)
    if args.output != "":
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
