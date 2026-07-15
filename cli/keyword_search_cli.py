import argparse
import collections
import json
import string
from nltk.stem import PorterStemmer
import pickle
import math


class InvertedIndex:
    index: dict[str, set[int]] = {}
    docmap: dict[int, dict] = {}
    term_frequencies: dict[int, collections.Counter] = {}

    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = preprocessing_and_tokenisation(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
        
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = collections.Counter()

        self.term_frequencies[doc_id].update(tokens)

    def get_tf(self, doc_id: int, token: str) -> int:
        return self.term_frequencies.get(doc_id, {}).get(token, 0)

    def get_documents(self, token: str) -> list[int]:
        doc_ids = self.index.get(token, set())
        return sorted(list(doc_ids))
    
    def build(self) -> None:
        data = load_movies()
        for movie in data:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self) -> None:
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)

            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            print("Cache files not found. Please run the 'build' command first.")
            exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Token to get term frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to get inverse document frequency for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    args = parser.parse_args()

    data = load_movies()
    stemmer = PorterStemmer()


    match args.command:
        case "search":
            invIdx = InvertedIndex()
            invIdx.load()
            print(f"Searching for: {args.query}")
            results = search(args.query, invIdx, limit=5)
                    
            if results:
                for i, movie in enumerate(results, start = 1):
                    print(f"{i}. {movie['title']}")

        case "build":
            build_command()

        case "tf":
            invIdx = InvertedIndex()
            invIdx.load()
            token = preprocessing_and_tokenisation(args.term)[0]
            tfVal = tf(args.doc_id, token, invIdx)
            print(f"Term frequency of '{args.term}' in document {args.doc_id}: {tfVal}")

        case "idf":
            invIdx = InvertedIndex()
            invIdx.load()
            idfVal = idf(args.term, invIdx)
            print(f"Inverse document frequency of '{args.term}': {idfVal:.2f}")

        case "tfidf":
            invIdx = InvertedIndex()
            invIdx.load()
            token = preprocessing_and_tokenisation(args.term)[0]
            tfVal = tf(args.doc_id, token, invIdx)
            idfVal = idf(args.term, invIdx)
            tfidfVal = tfVal * idfVal
            print(f"TF-IDF score of '{args.term}' in document {args.doc_id}: {tfidfVal:.2f}")

        case _:
            parser.print_help()


STOPWORDDS = []
with open("data/stopwords.txt", "r") as f:
        content = f.read().translate(str.maketrans("", "", string.punctuation)).lower()
        STOPWORDS = content.splitlines()


def search(query: str, invIdx: InvertedIndex, limit: int) -> list[dict]:
    cleanQuery = preprocessing_and_tokenisation(query)
    results_set = set()
    for q in cleanQuery:
        matching_doc_ids = invIdx.get_documents(q)
        for doc_id in matching_doc_ids:
            if (len(results_set) < limit):
                results_set.add(doc_id)

    results = [invIdx.docmap[doc_id] for doc_id in sorted(results_set)]
    return results


def tf(doc_id: int, token: str, invIdx: InvertedIndex) -> int:
    return invIdx.get_tf(doc_id, token)

def idf(term: str, invIdx: InvertedIndex) -> float:
    token = preprocessing_and_tokenisation(term)[0]
    totalDocCount = len(invIdx.docmap)
    docCountWithToken = len(invIdx.get_documents(token))
    idf = math.log((1 + totalDocCount) / (1 + docCountWithToken))
    return idf

def build_command() -> None:
    invIdx = InvertedIndex()
    invIdx.build()
    invIdx.save()
    # test: print(f"First document for token 'merida' = {invIdx.docmap[next(iter(invIdx.index.get('merida', set())))]['id']}")

def preprocessing_and_tokenisation(text: str) -> list[str]:
    clean_text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    splitted_text = clean_text.split()
    tokenized_text = [token for token in splitted_text if token not in STOPWORDS]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokenized_text]
    return stemmed_tokens

def load_movies() -> list[dict]:
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    return data["movies"]

if __name__ == "__main__":
    main()
