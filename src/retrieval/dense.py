"""Dense retriever backed by the original sentence-transformer implementation."""

from ._compat import load_movies
from .types import RetrievedDocument


class DenseRetriever:
    def __init__(self, documents: list[dict] | None = None) -> None:
        from lib.semantic_search import ChunkedSemanticSearch

        self.documents = documents if documents is not None else load_movies()
        self._by_id = {doc["id"]: doc for doc in self.documents}
        self.searcher = ChunkedSemanticSearch()
        self.searcher.load_or_create_chunk_embeddings(self.documents)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        # search_chunks truncates descriptions to 100 chars for CLI display;
        # downstream rerankers and generators need the full record.
        return [
            {
                "id": hit["id"],
                "title": self._by_id[hit["id"]]["title"],
                "description": self._by_id[hit["id"]]["description"],
                "score": hit["score"],
            }
            for hit in self.searcher.search_chunks(query, k)
        ]


def retrieve(query: str, k: int = 5) -> list[RetrievedDocument]:
    return DenseRetriever().retrieve(query, k)
