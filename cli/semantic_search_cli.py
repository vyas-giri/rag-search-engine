import argparse
from lib.semantic_search import embed_query_text, embed_text, search_command, verify_embeddings, verify_model

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the semantic search model")

    single_embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    single_embed_parser.add_argument("text", type=str, help="Text to generate embedding for")

    subparsers.add_parser("verify_embeddings", help="Verify the embeddings for the movie dataset")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate embedding for a given query text")
    embed_query_parser.add_argument("query", type=str, help="Query text to generate embedding for")

    search_parser = subparsers.add_parser("search", help="Search for similar documents based on a query")
    search_parser.add_argument("query", type=str, help="Query text to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results (default: 5)")


    args = parser.parse_args()


    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            search_command(args.query, args.limit)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()