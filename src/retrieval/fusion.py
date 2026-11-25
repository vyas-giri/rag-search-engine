"""Rank fusion utilities.

Reciprocal Rank Fusion (RRF) merges several ranked lists without looking at
their raw scores, which matters here because BM25 scores and cosine
similarities live on different, incomparable scales.

    rrf_score(doc) = sum over lists L containing doc of 1 / (k + rank_L(doc))

Documents found by only one retriever still receive a score (union semantics),
so a strong keyword-only or semantic-only match is not discarded.
"""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedResult:
    """A document id with its fused score and its 1-based rank in each source."""

    doc_id: Hashable
    score: float
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        return sorted(self.ranks)


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[Hashable]],
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[FusedResult]:
    """Fuse named ranked lists of document ids with RRF.

    Args:
        ranked_lists: mapping of source name (e.g. ``"bm25"``) to document ids
            ordered best-first. Repeated ids within one list keep their first
            (best) rank only.
        k: RRF smoothing constant. Larger values flatten the advantage of the
            top ranks. Must be positive.
        limit: maximum number of fused results to return.

    Returns:
        Results sorted by descending score. Ties are broken by the best rank
        any source gave the document, then by the order in which documents
        were first seen, so output is deterministic.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")

    ranks: dict[Hashable, dict[str, int]] = {}
    for source, doc_ids in ranked_lists.items():
        for rank, doc_id in enumerate(doc_ids, start=1):
            doc_ranks = ranks.setdefault(doc_id, {})
            doc_ranks.setdefault(source, rank)

    first_seen = {doc_id: order for order, doc_id in enumerate(ranks)}
    fused = [
        FusedResult(
            doc_id=doc_id,
            score=sum(1.0 / (k + rank) for rank in doc_ranks.values()),
            ranks=doc_ranks,
        )
        for doc_id, doc_ranks in ranks.items()
    ]
    fused.sort(
        key=lambda r: (-r.score, min(r.ranks.values()), first_seen[r.doc_id])
    )
    return fused if limit is None else fused[:limit]
