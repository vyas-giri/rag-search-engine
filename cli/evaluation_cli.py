import argparse
from lib.constants import GOLDEN_DATASET_PATH, K_PARAMETER, load_movies
from lib.hybrid_search import HybridSearch
import json

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    golden_dataset: list[dict] = json.load(open(GOLDEN_DATASET_PATH, "r"))["test_cases"]
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    for item in golden_dataset:
        query = item["query"]
        expected_results = item["relevant_docs"]
        rrf_results = hybrid_search.rrf_search(query, k=K_PARAMETER, limit=limit)
        relevant_retrieved = [res for res in rrf_results if res["title"] in expected_results]
        precision = len(relevant_retrieved) / len(rrf_results) if rrf_results else 0
        recall = len(relevant_retrieved) / len(expected_results) if expected_results else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        )
        print(f"k={limit}")
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1_score:.4f}")
        print(f"  - Retrieved: {[res['title'] for res in rrf_results]}")
        print(f"  - Relevant: {[res['title'] for res in relevant_retrieved]}")


if __name__ == "__main__":
     main()