"""Evaluate retrievers against the repository's golden dataset."""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from ._compat import load_movies
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .reranker import rerank

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "golden_dataset.json"


def _metrics(results: list[dict], relevant: list[str], k: int) -> dict[str, float]:
    retrieved = [doc.get("title", "") for doc in results[:k]]
    relevant_set = set(relevant)
    hits = [title for title in retrieved if title in relevant_set]
    precision = len(hits) / k if k else 0.0
    recall = len(set(hits)) / len(relevant_set) if relevant_set else 0.0
    reciprocal_rank = next(
        (1.0 / rank for rank, title in enumerate(retrieved, 1) if title in relevant_set),
        0.0,
    )
    dcg = sum(
        (1.0 if title in relevant_set else 0.0) / math.log2(rank + 1)
        for rank, title in enumerate(retrieved, 1)
    )
    ideal = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return {
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def evaluate(
    retriever_name: str = "hybrid", reranker_name: str = "none", k: int = 5
) -> dict[str, Any]:
    documents = load_movies()
    retrievers = {
        "bm25": BM25Retriever,
        "dense": DenseRetriever,
        "hybrid": HybridRetriever,
    }
    try:
        retriever = retrievers[retriever_name](documents)
    except KeyError as exc:
        raise ValueError(f"Unsupported retriever: {retriever_name}") from exc

    with DATASET_PATH.open() as dataset_file:
        cases = json.load(dataset_file)["test_cases"]
    queries = []
    for case in cases:
        started = time.perf_counter()
        results = retriever.retrieve(case["query"], k)
        results = rerank(results, case["query"], reranker_name, k)
        latency_ms = (time.perf_counter() - started) * 1000
        queries.append(
            {
                "query": case["query"],
                "retrieved": [doc.get("title", "") for doc in results],
                "latency_ms": latency_ms,
                **_metrics(results, case["relevant_docs"], k),
            }
        )
    metric_names = ("precision", "recall", "mrr", "ndcg", "latency_ms")
    aggregate = {
        name: sum(query[name] for query in queries) / len(queries) if queries else 0.0
        for name in metric_names
    }
    return {
        "retriever": retriever_name,
        "reranker": reranker_name,
        "k": k,
        "queries": queries,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="hybrid")
    parser.add_argument("--reranker", choices=("none", "cross_encoder"), default="none")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")
    print(json.dumps(evaluate(args.retriever, args.reranker, args.k), indent=2))


if __name__ == "__main__":
    main()
