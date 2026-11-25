"""Evaluate retrievers against the repository's golden dataset."""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from src.evaluation.retrieval import retrieval_metrics

from ._compat import load_movies
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .reranker import rerank
from .types import Retriever

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "golden_dataset.json"
DEFAULT_RERANK_CANDIDATES = 20
RETRIEVERS = ("bm25", "dense", "hybrid", "qdrant", "hybrid_qdrant")


def build_retriever(name: str, documents: list[dict]) -> Retriever:
    """Construct a retriever by name. Qdrant is imported only when requested."""
    if name == "bm25":
        return BM25Retriever(documents)
    if name == "dense":
        return DenseRetriever(documents)
    if name == "hybrid":
        return HybridRetriever(documents)
    if name in ("qdrant", "hybrid_qdrant"):
        from .vector_store import QdrantDenseRetriever

        qdrant = QdrantDenseRetriever(documents)
        return qdrant if name == "qdrant" else HybridRetriever(documents, dense=qdrant)
    raise ValueError(f"Unsupported retriever: {name}")


def evaluate(
    retriever_name: str = "hybrid",
    reranker_name: str = "none",
    k: int = 5,
    candidates: int | None = None,
) -> dict[str, Any]:
    """Run every golden query and report per-query and mean metrics.

    ``candidates`` is how many documents the retriever returns before
    reranking. It defaults to ``k`` without a reranker and to
    ``max(k, 20)`` with one; reranking only ``k`` documents could reorder
    them but never surface anything new.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if candidates is None:
        candidates = k if reranker_name == "none" else max(k, DEFAULT_RERANK_CANDIDATES)
    if candidates < k:
        raise ValueError(f"candidates ({candidates}) must be >= k ({k})")

    documents = load_movies()
    retriever = build_retriever(retriever_name, documents)

    with DATASET_PATH.open() as dataset_file:
        cases = json.load(dataset_file)["test_cases"]
    # Untimed warm-up so one-off costs (lazy model init, cold index pages)
    # are not charged to the first golden query.
    if cases:
        rerank(retriever.retrieve(cases[0]["query"], candidates), cases[0]["query"], reranker_name, k)

    queries = []
    for case in cases:
        started = time.perf_counter()
        results = retriever.retrieve(case["query"], candidates)
        results = rerank(results, case["query"], reranker_name, k)
        latency_ms = (time.perf_counter() - started) * 1000
        titles = [doc.get("title", "") for doc in results]
        queries.append(
            {
                "query": case["query"],
                "retrieved": titles,
                "latency_ms": latency_ms,
                **retrieval_metrics(titles, case["relevant_docs"], k),
            }
        )
    metric_names = ("precision", "recall", "mrr", "ndcg", "latency_ms")
    aggregate = {
        name: sum(query[name] for query in queries) / len(queries) if queries else 0.0
        for name in metric_names
    }
    aggregate["latency_p50_ms"] = (
        statistics.median(query["latency_ms"] for query in queries) if queries else 0.0
    )
    return {
        "retriever": retriever_name,
        "reranker": reranker_name,
        "k": k,
        "candidates": candidates,
        "queries": queries,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    parser.add_argument("--reranker", choices=("none", "cross_encoder"), default="none")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="documents retrieved before reranking (default: k, or max(k, 20) with a reranker)",
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.candidates is not None and args.candidates < args.k:
        parser.error("--candidates must be >= --k")
    print(json.dumps(evaluate(args.retriever, args.reranker, args.k, args.candidates), indent=2))


if __name__ == "__main__":
    main()
