from typing import cast
from sentence_transformers import SentenceTransformer
from torch import Tensor
import numpy as np
import os
from lib.constants import CACHE_PATH
from lib.keyword_search import load_movies


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str) -> Tensor:
        if text is None or text.strip() == "":
            raise ValueError("Input text cannot be empty or None.")
        return self.model.encode(text)
    
    def build_embeddings(self, documents: list[dict]) -> Tensor:
        self.documents = documents
        movie_strs = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            movie_strs.append(f"{doc['title']}: {doc['description']}")
        
        self.embeddings = self.model.encode(movie_strs, show_progress_bar=True)

        with open(os.path.join(CACHE_PATH, "movie_embeddings.npy"), "wb") as f:
            np.save(f, self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]) -> Tensor:
        self.documents = documents
        if os.path.exists(os.path.join(CACHE_PATH, "movie_embeddings.npy")):
            self.embeddings = np.load(os.path.join(CACHE_PATH, "movie_embeddings.npy"))
            if len(self.embeddings) == len(documents):
                return cast(Tensor, self.embeddings)
            else:
                print("Embeddings count does not match documents count. Rebuilding embeddings.")
                return self.build_embeddings(documents)
        else:
            return self.build_embeddings(documents)
        
    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        embedded_query = self.generate_embedding(query)
        howSimilar = []
        for doc, embedding in zip(self.documents, self.embeddings):
            similarity = cosine_similarity(embedded_query, embedding)
            howSimilar.append((similarity, doc))

        # Sort by similarity (descending) and return the top 'limit' results
        howSimilar.sort(key=lambda x: x[0], reverse=True)
        res = []
        for i in range(min(limit, len(howSimilar))):
            result = {
                "score": howSimilar[i][0],
                "title": howSimilar[i][1]['title'],
                "description": howSimilar[i][1]['description'],
            }
            res.append(result)
        return res


def verify_model():
    try:
        ss = SemanticSearch()
        print(f"Model loaded: {ss.model}")
        print(f"Max sequence length: {ss.model.max_seq_length}")
    except Exception as e:
        print(f"Error occurred: {e}")

def embed_text(text: str):
    ss = SemanticSearch()
    embedding  = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")   

def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: Tensor, vec2: Tensor) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0  # Avoid division by zero; return 0 similarity if either vector is zero
    
    return dot_product / (norm1 * norm2)

def search_command(query: str, limit: int):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    res = ss.search(query, limit)
    for i, result in enumerate(res):
        print(f"{i + 1}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   {result['description']}\n")