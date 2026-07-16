import math
import collections
from nltk.stem import PorterStemmer
from lib.constants import BM25_B, BM25_K1, CACHE_PATH, STOPWORDS, DATA_PATH
import pickle
import math
import json
import string
import os


class InvertedIndex:
    index: dict[str, set[int]] = {}
    docmap: dict[int, dict] = {}
    term_frequencies: dict[int, collections.Counter] = {}
    doc_lengths: dict[int, int] = {}

    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = preprocessing_and_tokenisation(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
        
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = collections.Counter()

        self.term_frequencies[doc_id].update(tokens)
    
    def build(self) -> None:
        data = load_movies()
        for movie in data:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self) -> None:
        with open(os.path.join(CACHE_PATH, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        
        with open(os.path.join(CACHE_PATH, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        
        with open(os.path.join(CACHE_PATH, "doc_lengths.pkl"), "wb") as f:
            pickle.dump(self.doc_lengths, f)
        
        with open(os.path.join(CACHE_PATH, "term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        try:
            with open(os.path.join(CACHE_PATH, "index.pkl"), "rb") as f:
                self.index = pickle.load(f)
            
            with open(os.path.join(CACHE_PATH, "docmap.pkl"), "rb") as f:
                self.docmap = pickle.load(f)

            with open(os.path.join(CACHE_PATH, "term_frequencies.pkl"), "rb") as f:
                self.term_frequencies = pickle.load(f)
            
            with open(os.path.join(CACHE_PATH, "doc_lengths.pkl"), "rb") as f:
                self.doc_lengths = pickle.load(f)
        except FileNotFoundError:
            print("Cache files not found. Please run the 'build' command first.")
            exit(1)
        
    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths) if self.doc_lengths else 0.0

    def get_tf(self, doc_id: int, token: str) -> int:
        return self.term_frequencies.get(doc_id, {}).get(token, 0)
    
    def get_bm25_idf(self, token: str) -> float:
        totalDocCount = len(self.docmap)
        docCountWithToken = len(self.get_documents(token))
        bmidf = math.log((totalDocCount - docCountWithToken + 0.5) / (docCountWithToken + 0.5) + 1)
        return bmidf
    
    def get_bm25_tf(self, doc_id: int, token: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        length_normalisation = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1
        tf = self.get_tf(doc_id, token)
        return (tf * (k1 + 1)) / (tf + k1 * length_normalisation) if length_normalisation > 0 else 1
    
    def bm25_score(self, doc_id: int, token: str) -> float:
        bm25idf = self.get_bm25_idf(token)
        bm25tf = self.get_bm25_tf(doc_id, token)
        return bm25idf * bm25tf
    
    def bm25_search(self, query: str, limit: int) -> list[dict]:
        tokenized_query = preprocessing_and_tokenisation(query)
        scores = {}
        for doc_id in self.docmap.keys():
            score = sum(self.bm25_score(doc_id, token) for token in tokenized_query)
            scores[doc_id] = score
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        res = []
        for doc_id, score in sorted_scores[:limit]:
            movie = self.docmap[doc_id]
            movie['bm25_score'] = score
            res.append(movie)
        return res

    def get_documents(self, token: str) -> list[int]:
        doc_ids = self.index.get(token, set())
        return sorted(list(doc_ids))


def search(query: str, limit: int) -> list[dict]:
    invIdx = InvertedIndex()
    invIdx.load()
    cleanQuery = preprocessing_and_tokenisation(query)
    results_set = set()
    for q in cleanQuery:
        matching_doc_ids = invIdx.get_documents(q)
        for doc_id in matching_doc_ids:
            if (len(results_set) < limit):
                results_set.add(doc_id)

    results = [invIdx.docmap[doc_id] for doc_id in sorted(results_set)]
    return results


def tf(doc_id: int, term: str) -> int:
    invIdx = InvertedIndex()
    invIdx.load()
    token = preprocessing_and_tokenisation(term)[0]
    return invIdx.get_tf(doc_id, token)

def idf(term: str) -> float:
    invIdx = InvertedIndex()
    invIdx.load()
    token = preprocessing_and_tokenisation(term)[0]
    totalDocCount = len(invIdx.docmap)
    docCountWithToken = len(invIdx.get_documents(token))
    idf = math.log((1 + totalDocCount) / (1 + docCountWithToken))
    return idf

def bm25_idf_command(term: str) -> float:
    invIdx = InvertedIndex()
    invIdx.load()
    token = preprocessing_and_tokenisation(term)[0]
    bmidf = invIdx.get_bm25_idf(token)
    return bmidf

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    invIdx = InvertedIndex()
    invIdx.load()
    token = preprocessing_and_tokenisation(term)[0]
    return invIdx.get_bm25_tf(doc_id, token, k1, b)

def bm25_search_command(query: str, limit: int = 5) -> list[dict]:
    invIdx = InvertedIndex()
    invIdx.load()
    return invIdx.bm25_search(query, limit)

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
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]