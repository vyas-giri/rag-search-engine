"""Ranking metrics for retrieval evaluation.

All functions take the retrieved titles in rank order (best first) and the set
of relevant titles. Relevance is binary. A relevant title that appears more
than once in the results only counts at its first (best) position, so
duplicate results cannot inflate a score.
"""

import math
from collections.abc import Collection, Sequence


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


def _first_hit_positions(retrieved: Sequence[str], relevant: Collection[str], k: int) -> list[int]:
    """1-based positions within the top k of the first occurrence of each relevant item."""
    seen: set[str] = set()
    positions = []
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant and item not in seen:
            seen.add(item)
            positions.append(rank)
    return positions


def precision_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of the k slots filled by distinct relevant items."""
    _validate_k(k)
    return len(_first_hit_positions(retrieved, relevant, k)) / k


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of relevant items found in the top k. 0.0 when nothing is relevant."""
    _validate_k(k)
    if not relevant:
        return 0.0
    return len(_first_hit_positions(retrieved, relevant, k)) / len(set(relevant))


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1 / position of the first relevant item in the top k, else 0.0."""
    _validate_k(k)
    positions = _first_hit_positions(retrieved, relevant, k)
    return 1.0 / positions[0] if positions else 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Binary-relevance nDCG@k. 0.0 when nothing is relevant."""
    _validate_k(k)
    ideal_hits = min(k, len(set(relevant)))
    if ideal_hits == 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in _first_hit_positions(retrieved, relevant, k))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def retrieval_metrics(
    retrieved: Sequence[str], relevant: Collection[str], k: int
) -> dict[str, float]:
    """All ranking metrics for one query."""
    relevant_set = set(relevant)
    return {
        "precision": precision_at_k(retrieved, relevant_set, k),
        "recall": recall_at_k(retrieved, relevant_set, k),
        "mrr": reciprocal_rank(retrieved, relevant_set, k),
        "ndcg": ndcg_at_k(retrieved, relevant_set, k),
    }
