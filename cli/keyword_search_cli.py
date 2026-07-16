import argparse
from lib.constants import BM25_B, BM25_K1
from lib.keyword_search import bm25_tf_command, search, build_command, tf, idf, bm25_idf_command, bm25_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Token to get term frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to get inverse document frequency for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 B parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using BM25 ranking")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to return")

    args = parser.parse_args()


    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search(args.query, limit=5)
                    
            if results:
                for i, movie in enumerate(results, start = 1):
                    print(f"{i}. {movie['title']}")

        case "build":
            build_command()

        case "tf":
            tfVal = tf(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document {args.doc_id}: {tfVal}")

        case "idf":
            idfVal = idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idfVal:.2f}")
        
        case "bm25idf":
            bmidfVal = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bmidfVal:.2f}")

        case "bm25tf":
            bm25tfVal = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tfVal:.2f}")

        case "tfidf":
            tfVal = tf(args.doc_id, args.term)
            idfVal = idf(args.term)
            tfidfVal = tfVal * idfVal
            print(f"TF-IDF score of '{args.term}' in document {args.doc_id}: {tfidfVal:.2f}")

        case "bm25search":
            res = bm25_search_command(args.query, args.limit)
            if res:
                for i, movie in enumerate(res, start = 1):
                    print(f"{i}. ({movie['id']}) {movie['title']} - BM25 Score: {movie['bm25_score']:.2f}")
            else:
                print("No results found.")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
