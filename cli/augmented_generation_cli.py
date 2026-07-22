import argparse
from lib.augmented_generation import rag_command, summarize_command, citations_command, question_command
from lib.constants import DEFAULT_QUERY_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Search and summarize results")
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT, help="Limit the number of search results to summarize")

    citations_parser = subparsers.add_parser("citations", help="Search and provide citations for results")
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT, help="Limit the number of search results for citations")

    question_parser = subparsers.add_parser("question", help="Answer a question based on search results")
    question_parser.add_argument("query", type=str, help="Question to answer based on search results")
    question_parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT, help="Limit the number of search results to consider for answering the question")

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_command(args.query)

        case "summarize":
            summarize_command(args.query, args.limit)

        case "citations":
            citations_command(args.query, args.limit)

        case "question":
            question_command(args.query, args.limit)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()