"""Qdrant vector store tests using the in-process ``:memory:`` client (no server)."""

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import (
    QdrantDenseRetriever,
    ensure_collection,
    upload_chunks,
)

DIM = 4
DOCS = [
    {"id": 10, "title": "Space Movie", "description": "Astronauts alone in orbit."},
    {"id": 20, "title": "Bear Movie", "description": "A bear eats marmalade."},
    {"id": 30, "title": "Car Movie", "description": "Fast cars race."},
]
# Two chunks for movie 10, one each for 20 and 30. Axis 0 = "space", 1 = "bear", 2 = "cars".
VECTORS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],  # movie 10, chunk 0: very "space"
        [0.9, 0.1, 0.0, 0.0],  # movie 10, chunk 1: also "space"
        [0.6, 0.8, 0.0, 0.0],  # movie 20: mostly "bear", a bit "space"
        [0.0, 0.0, 1.0, 0.0],  # movie 30: "cars"
    ],
    dtype=np.float32,
)
METADATA = [
    {"movie_idx": 10, "chunk_idx": 0},
    {"movie_idx": 10, "chunk_idx": 1},
    {"movie_idx": 20, "chunk_idx": 0},
    {"movie_idx": 30, "chunk_idx": 0},
]
QUERIES = {"space": [1.0, 0.0, 0.0, 0.0], "bear": [0.0, 1.0, 0.0, 0.0]}


@pytest.fixture
def client():
    client = QdrantClient(":memory:")
    ensure_collection(client, dim=DIM)
    upload_chunks(client, VECTORS, METADATA, {d["id"]: d["title"] for d in DOCS})
    return client


@pytest.fixture
def retriever(client):
    return QdrantDenseRetriever(documents=DOCS, client=client, encoder=QUERIES.__getitem__)


def test_upload_creates_one_point_per_chunk(client):
    assert client.count("movie_chunks").count == len(METADATA)


def test_ensure_collection_is_idempotent(client):
    ensure_collection(client, dim=DIM)
    assert client.count("movie_chunks").count == len(METADATA)


def test_recreate_drops_points(client):
    ensure_collection(client, dim=DIM, recreate=True)
    assert client.count("movie_chunks").count == 0


def test_upload_rejects_mismatched_metadata(client):
    with pytest.raises(ValueError):
        upload_chunks(client, VECTORS, METADATA[:2], {})


def test_results_are_grouped_one_per_movie(retriever):
    results = retriever.retrieve("space", k=3)
    ids = [doc["id"] for doc in results]
    assert ids == [10, 20, 30][: len(ids)]
    assert len(ids) == len(set(ids))


def test_returns_full_records_with_score(retriever):
    top = retriever.retrieve("bear", k=1)[0]
    assert top["id"] == 20
    assert top["description"] == "A bear eats marmalade."
    assert top["score"] == pytest.approx(0.8, abs=1e-4)


def test_respects_k(retriever):
    assert len(retriever.retrieve("space", k=1)) == 1


def test_blank_query_returns_nothing(retriever):
    assert retriever.retrieve("   ", k=3) == []


def test_invalid_k(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("space", k=0)


def test_missing_collection_has_actionable_error():
    with pytest.raises(RuntimeError, match="src.retrieval.ingest"):
        QdrantDenseRetriever(documents=DOCS, client=QdrantClient(":memory:"), encoder=QUERIES.__getitem__)


def test_plugs_into_hybrid_retriever(retriever):
    class FakeBM25:
        def retrieve(self, query, k=5):
            return [{"id": 30}]

    results = HybridRetriever(documents=DOCS, bm25=FakeBM25(), dense=retriever).retrieve("space", k=3)
    by_id = {doc["id"]: doc for doc in results}
    assert by_id[10]["sources"] == ["dense"]
    assert by_id[30]["sources"] == ["bm25", "dense"]
