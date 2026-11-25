"""Shared types for retrieval implementations."""

from typing import Protocol, TypedDict


class RetrievedDocument(TypedDict, total=False):
    """A document returned by a retriever.

    The source data always contains ``id``, ``title`` and ``description``.
    Retrieval implementations may add scores and rank-specific metadata.
    """

    id: int
    title: str
    description: str
    score: float
    bm25Score: float
    semanticScore: float
    hybridScore: float
    rrfScore: float


class Retriever(Protocol):
    """Protocol implemented by all query retrievers."""

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        """Return up to ``k`` documents ranked for ``query``."""
