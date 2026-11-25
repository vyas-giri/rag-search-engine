"""Qdrant-backed dense retrieval over movie description chunks.

One Qdrant point per chunk (same chunks and all-MiniLM-L6-v2 embeddings as the
in-process ``DenseRetriever``), with the owning movie id in the payload.
Queries group hits by ``movie_id`` server-side, so each movie is scored by its
best-matching chunk.
"""

import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from ._compat import load_movies
from .types import RetrievedDocument

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache"
COLLECTION = "movie_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
# HNSW candidate-list size at query time. Measured on the golden queries at
# depth 100: default ~0.88 overlap with exact search, 512 ~0.99 (+~8 ms).
DEFAULT_HNSW_EF = 512

Encoder = Callable[[str], Sequence[float]]


def get_client(url: str | None = None) -> QdrantClient:
    """Connect to Qdrant at ``url``, ``$QDRANT_URL``, or localhost."""
    return QdrantClient(url=url or os.environ.get("QDRANT_URL", "http://localhost:6333"))


def ensure_collection(
    client: QdrantClient,
    collection: str = COLLECTION,
    dim: int = EMBEDDING_DIM,
    recreate: bool = False,
) -> None:
    """Create the chunk collection and its ``movie_id`` payload index if missing."""
    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    # Grouping by movie_id is on the hot path of every query.
    client.create_payload_index(collection, "movie_id", models.PayloadSchemaType.INTEGER)


def upload_chunks(
    client: QdrantClient,
    vectors: np.ndarray,
    chunk_metadata: list[dict],
    titles: dict[int, str],
    collection: str = COLLECTION,
    batch_size: int = 512,
) -> None:
    """Upsert one point per chunk. Point ids are chunk positions, so reruns overwrite."""
    if len(vectors) != len(chunk_metadata):
        raise ValueError(
            f"{len(vectors)} vectors but {len(chunk_metadata)} metadata rows"
        )
    client.upload_collection(
        collection,
        vectors=vectors,
        payload=(
            {
                "movie_id": meta["movie_idx"],
                "chunk_idx": meta["chunk_idx"],
                "title": titles[meta["movie_idx"]],
            }
            for meta in chunk_metadata
        ),
        ids=range(len(chunk_metadata)),
        batch_size=batch_size,
        wait=True,
    )


def wait_until_indexed(
    client: QdrantClient, collection: str = COLLECTION, timeout_s: float = 600
) -> None:
    """Block until Qdrant's optimizer has finished building the HNSW index."""
    deadline = time.monotonic() + timeout_s
    while client.get_collection(collection).status != models.CollectionStatus.GREEN:
        if time.monotonic() > deadline:
            raise TimeoutError(f"collection {collection!r} not indexed after {timeout_s}s")
        time.sleep(1)


def load_cached_chunks(cache_dir: Path = CACHE_DIR) -> tuple[np.ndarray, list[dict]]:
    """Load the chunk embeddings and metadata built by ``ChunkedSemanticSearch``."""
    vectors_path = cache_dir / "chunk_embeddings.npy"
    metadata_path = cache_dir / "chunk_metadata.json"
    if not vectors_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {vectors_path.name} or {metadata_path.name} in {cache_dir}. "
            "Build them with: uv run python cli/semantic_search_cli.py embed_chunks"
        )
    vectors = np.load(vectors_path)
    with metadata_path.open() as f:
        metadata = json.load(f)["chunks"]
    return vectors, metadata


class QdrantDenseRetriever:
    """Dense retriever whose nearest-neighbour search runs in Qdrant."""

    def __init__(
        self,
        documents: list[dict] | None = None,
        client: QdrantClient | None = None,
        encoder: Encoder | None = None,
        collection: str = COLLECTION,
        exact: bool = False,
        hnsw_ef: int = DEFAULT_HNSW_EF,
    ) -> None:
        self.documents = documents if documents is not None else load_movies()
        self._by_id = {doc["id"]: doc for doc in self.documents}
        self.client = client if client is not None else get_client()
        self.collection = collection
        self.search_params = models.SearchParams(exact=exact, hnsw_ef=hnsw_ef)
        if not self.client.collection_exists(collection):
            raise RuntimeError(
                f"Qdrant collection {collection!r} not found. "
                "Run: uv run python -m src.retrieval.ingest"
            )
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(EMBEDDING_MODEL)
            encoder = model.encode
        self.encode = encoder

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if not query.strip():
            return []
        response = self.client.query_points_groups(
            self.collection,
            query=list(map(float, self.encode(query))),
            group_by="movie_id",
            group_size=1,
            limit=k,
            search_params=self.search_params,
            with_payload=False,
        )
        results: list[RetrievedDocument] = []
        for group in response.groups:
            doc = self._by_id[group.id]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "score": group.hits[0].score,
                }
            )
        return results
