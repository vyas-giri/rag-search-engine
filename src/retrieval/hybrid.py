"""Hybrid retriever: BM25 + dense candidates merged with Reciprocal Rank Fusion."""

from ._compat import load_movies
from .fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from .types import RetrievedDocument, Retriever

DEFAULT_CANDIDATES = 100


class HybridRetriever:
    """Fuse a keyword and a semantic retriever with union RRF.

    Each underlying retriever returns its top ``candidates`` documents; the two
    rankings are fused and the top ``k`` returned. Retrievers can be injected,
    which keeps this class testable without loading models or indexes.
    """

    def __init__(
        self,
        documents: list[dict] | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        candidates: int = DEFAULT_CANDIDATES,
        bm25: Retriever | None = None,
        dense: Retriever | None = None,
    ) -> None:
        if candidates < 1:
            raise ValueError(f"candidates must be at least 1, got {candidates}")
        self.documents = documents if documents is not None else load_movies()
        self.rrf_k = rrf_k
        self.candidates = candidates
        self._by_id = {doc["id"]: doc for doc in self.documents}

        if bm25 is None:
            from .bm25 import BM25Retriever

            bm25 = BM25Retriever(self.documents)
        if dense is None:
            from .dense import DenseRetriever

            dense = DenseRetriever(self.documents)
        self.bm25 = bm25
        self.dense = dense

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        pool = max(self.candidates, k)
        fused = reciprocal_rank_fusion(
            {
                "bm25": [doc["id"] for doc in self.bm25.retrieve(query, pool)],
                "dense": [doc["id"] for doc in self.dense.retrieve(query, pool)],
            },
            k=self.rrf_k,
            limit=k,
        )
        results: list[RetrievedDocument] = []
        for item in fused:
            source = self._by_id[item.doc_id]
            results.append(
                {
                    "id": source["id"],
                    "title": source["title"],
                    "description": source["description"],
                    "score": item.score,
                    "rrfScore": item.score,
                    "bm25Rank": item.ranks.get("bm25"),
                    "semanticRank": item.ranks.get("dense"),
                    "sources": item.sources,
                }
            )
        return results


def retrieve(query: str, k: int = 5) -> list[RetrievedDocument]:
    return HybridRetriever().retrieve(query, k)
