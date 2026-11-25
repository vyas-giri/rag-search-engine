"""Load cached chunk embeddings into Qdrant.

    uv run python -m src.retrieval.ingest [--recreate] [--url URL]

Reuses ``cache/chunk_embeddings.npy`` instead of re-embedding, so the vectors
in Qdrant are identical to the in-process dense baseline.
"""

import argparse
import time

from ._compat import load_movies
from .vector_store import (
    COLLECTION,
    ensure_collection,
    get_client,
    load_cached_chunks,
    upload_chunks,
    wait_until_indexed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest movie chunks into Qdrant.")
    parser.add_argument("--url", default=None, help="Qdrant URL (default: $QDRANT_URL or localhost)")
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    args = parser.parse_args()

    client = get_client(args.url)
    vectors, metadata = load_cached_chunks()
    titles = {movie["id"]: movie["title"] for movie in load_movies()}

    ensure_collection(client, dim=vectors.shape[1], recreate=args.recreate)
    started = time.perf_counter()
    upload_chunks(client, vectors, metadata, titles)
    uploaded = time.perf_counter() - started
    print(f"Uploaded {len(metadata):,} chunks in {uploaded:.1f}s; waiting for index...")
    wait_until_indexed(client)
    info = client.get_collection(COLLECTION)
    print(
        f"Collection {COLLECTION!r}: {info.points_count:,} points, status {info.status.value}, "
        f"ready after {time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
