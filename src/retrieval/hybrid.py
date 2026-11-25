"""Hybrid retriever backed by the original RRF implementation."""

from ._compat import load_movies
from .types import RetrievedDocument


class HybridRetriever:
    def __init__(self, documents: list[dict] | None = None, rrf_k: int = 60) -> None:
        from lib.hybrid_search import HybridSearch

        self.documents = documents if documents is not None else load_movies()
        self.rrf_k = rrf_k
        self.searcher = HybridSearch(self.documents)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        return self.searcher.rrf_search(query, k=self.rrf_k, limit=k)


def retrieve(query: str, k: int = 5) -> list[RetrievedDocument]:
    return HybridRetriever().retrieve(query, k)
