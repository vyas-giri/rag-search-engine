import argparse
from lib.hybrid_search import normalize_command, weighted_search_command, rrf_search_command
from lib.constants import ALPHA, DEFAULT_QUERY_LIMIT, K_PARAMETER

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the input data")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="List of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=ALPHA, help="Weighting factor for the hybrid search")
    weighted_search_parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT, help="Limit the number of results returned")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform RRF hybrid search (not implemented)")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", type=int, default=K_PARAMETER, help="RRF parameter k")
    rrf_search_parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT, help="Limit the number of results returned")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement option (e.g., spell correction)")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Reranking method to use (e.g., individual)")
    rrf_search_parser.add_argument("--evaluate", action="store_true", help="Evaluate the search results against an LLM")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            if args.scores:
                normalize_command(args.scores)

        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)

        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit, args.enhance, args.rerank_method, args.evaluate)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()