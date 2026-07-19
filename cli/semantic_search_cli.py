import argparse
from lib.semantic_search import (
    embed_query_text, 
    embed_text, search_command, 
    verify_embeddings, 
    verify_model, 
    chunk_command, 
    semantic_chunk_text,
    embed_chunks_command,
    search_chunked_command
)
from lib.constants import DEFAULT_SEMANTIC_CHUNK_SIZE

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

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a given text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk (default: 200 words)")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks (default: 0)")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a given text into smaller pieces based on semantic similarity")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_SEMANTIC_CHUNK_SIZE, help="Maximum size of each chunk (default: 4 words)")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks (default: 0)")

    embed_chunk_parser = subparsers.add_parser("embed_chunks", help="Generate embeddings for all the chunks of all documents in the dataset")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for similar chunks based on a query")
    search_chunked_parser.add_argument("query", type=str, help="Query text to search for")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results (default: 5)")


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

        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)

        case "embed_chunks":
            embed_chunks_command()

        case "search_chunked":
            search_chunked_command(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()