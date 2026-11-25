"""BM25 retriever backed by the original inverted-index implementation."""

from ._compat import load_movies
from .types import RetrievedDocument


class BM25Retriever:
    def __init__(self, documents: list[dict] | None = None) -> None:
        from lib.keyword_search import InvertedIndex

        self.documents = documents if documents is not None else load_movies()
        self.index = InvertedIndex()
        self.index.load()

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        return self.index.bm25_search(query, k)


def retrieve(query: str, k: int = 5) -> list[RetrievedDocument]:
    return BM25Retriever().retrieve(query, k)
