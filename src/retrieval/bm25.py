"""BM25 retriever backed by the original inverted-index implementation."""

import os

from ._compat import load_movies
from .types import RetrievedDocument


class BM25Retriever:
    def __init__(self, documents: list[dict] | None = None) -> None:
        from lib.keyword_search import InvertedIndex

        self.documents = documents if documents is not None else load_movies()
        self.index = InvertedIndex()
        if not os.path.exists(self.index.index_path):
            self.index.build()
            self.index.save()
        self.index.load()

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        # bm25_search attaches the score to the index's own dicts; copy so
        # callers never share (or mutate) index state.
        return [dict(doc, score=doc["bm25_score"]) for doc in self.index.bm25_search(query, k)]


def retrieve(query: str, k: int = 5) -> list[RetrievedDocument]:
    return BM25Retriever().retrieve(query, k)
