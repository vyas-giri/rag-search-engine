import argparse
import json
import string
from nltk.stem import PorterStemmer


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    data = {}
    stopwords = []
    stemmer = PorterStemmer()

    with open("data/stopwords.txt", "r") as f:
        content = f.read().translate(str.maketrans("", "", string.punctuation)).lower()
        stopwords = content.splitlines()

    

    with open("data/movies.json", "r") as f:
        data = json.load(f)


    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            cleanQuery = preprocessing(args.query)
            cleanQuery = [token for token in cleanQuery if token not in stopwords]
            cleanQuery = [stemmer.stem(token) for token in cleanQuery]
            results = []
            for movie in data["movies"]:
                movieTitle = preprocessing(movie["title"])
                hasCommon = any(query in title for title in movieTitle for query in cleanQuery)
                if hasCommon:
                    results.append(movie)
            if results:
                for i, movie in enumerate(results[:5], start = 1):
                    print(f"{i}. {movie['title']}")
        case _:
            parser.print_help()


def preprocessing(text: str) -> list[str]:
    clean_text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    tokens = clean_text.split()
    return tokens

if __name__ == "__main__":
    main()
