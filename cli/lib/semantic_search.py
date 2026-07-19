import math
from typing import cast
from sentence_transformers import SentenceTransformer
from torch import Tensor
import numpy as np
import os
from lib.constants import CACHE_PATH, SCORE_PRECISION
from lib.keyword_search import load_movies
import re
import json
from lib.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_SEMANTIC_CHUNK_SIZE


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
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
    

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> Tensor:
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        allChunks = []
        metadata: list[dict] = []
        for doc in documents:
            if doc['description']:
                chunks = semantic_chunk(doc['description'], max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
                allChunks.extend(chunks)
                for i in range(len(chunks)):
                    metadata.append({
                        "movie_idx": doc['id'],
                        "chunk_idx": i,
                        "total_chunks": len(chunks),
                    })
                
        self.chunk_embeddings = self.model.encode(allChunks, show_progress_bar=True)
        self.chunk_metadata = metadata

        with open(os.path.join(CACHE_PATH, "chunk_embeddings.npy"), "wb") as f:
            np.save(f, self.chunk_embeddings)
        
        with open(os.path.join(CACHE_PATH, "chunk_metadata.json"), "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(allChunks)}, f, indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> Tensor:
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        if os.path.exists(os.path.join(CACHE_PATH, "chunk_embeddings.npy")) and os.path.exists(os.path.join(CACHE_PATH, "chunk_metadata.json")):
            self.chunk_embeddings = np.load(os.path.join(CACHE_PATH, "chunk_embeddings.npy"))
            with open(os.path.join(CACHE_PATH, "chunk_metadata.json"), "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            if len(self.chunk_embeddings) == metadata["total_chunks"]:
                return self.chunk_embeddings
            else:
                print("Chunk embeddings count does not match metadata count. Rebuilding chunk embeddings.")
                return self.build_chunk_embeddings(documents)
        else:
            return self.build_chunk_embeddings(documents)
        
    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        
        embeddedQuery = self.generate_embedding(query)
        best_scores: dict[int, float] = {}

        for embedding, meta in zip(self.chunk_embeddings, self.chunk_metadata):
            similarity = cosine_similarity(embeddedQuery, embedding)
            movie_idx = meta["movie_idx"]

            current_best = best_scores.get(movie_idx)
            if current_best is None or similarity > current_best:
                best_scores[movie_idx] = similarity
            
        ranked_movies = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for movie_idx, score in ranked_movies[:limit]:
            movie = self.document_map[movie_idx]
            results.append({
                "id": movie_idx,
                "title": movie["title"],
                "description": movie["description"][:100],
                "score": round(score, SCORE_PRECISION),
            })
        
        return results

                

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

def chunk_command(text: str, chunk_size: int, overlap: int):
    print(f"Chunking {len(text)} characters")
    txt = text.split()
    step = chunk_size - overlap
    idx = 1
    for i in range(0, len(txt), step):
        chunk = " ".join(txt[i:i+chunk_size])
        if chunk:
            print(f"{idx}. {chunk}")
            idx += 1

def semantic_chunk(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]
    
    i = 0
    chunks: list[str] = []
    while i < len(sentences):
        chunk_sentences = sentences[i:i + max_chunk_size]
        if chunk_sentences and len(chunk_sentences) <= overlap:
            break

        cleaned_sentences = []
        for chunk_sentence in chunk_sentences:
            chunk_sentence = chunk_sentence.strip()
            if chunk_sentence:
                cleaned_sentences.append(chunk_sentence)
        if not cleaned_sentences:
            i += max_chunk_size - overlap
            continue
        chunk = " ".join(cleaned_sentences)
        chunks.append(chunk)
        i += max_chunk_size - overlap
    return chunks

def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def embed_chunks_command():
    ss = ChunkedSemanticSearch()
    movies = load_movies()
    embeddings = ss.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked_command(query: str, limit: int):
    ss = ChunkedSemanticSearch()
    movies = load_movies()
    ss.load_or_create_chunk_embeddings(movies)
    res = ss.search_chunks(query, limit)
    for i, result in enumerate(res, start=1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description']}...")