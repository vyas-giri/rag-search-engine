from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.constants import load_movies, CACHE_PATH
from lib.semantic_search import cosine_similarity
import os
import numpy as np

class MultimodalSearch:
    def __init__(self, documents: list[dict] | None, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents if documents is not None else []
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        if not (os.path.exists(os.path.join(CACHE_PATH, "text_embeddings.npy"))):
            self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
            np.save(os.path.join(CACHE_PATH, "text_embeddings.npy"), self.text_embeddings)
        else:
            self.text_embeddings = np.load(os.path.join(CACHE_PATH, "text_embeddings.npy"))

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embedding = self.model.encode([image])
        return embedding[0]

    def search_with_image(self, image_path: str) -> list[dict]:
        image_embedding = self.embed_image(image_path)
        
        similarities: list[tuple[int, float]] = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for idx, score in similarities[:5]:
            doc = self.documents[idx]
            results.append({"title": doc["title"], "description": doc["description"], "score": score})

        return results


def verify_image_emmbedding(image_path: str) -> None:
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path: str) -> list[dict]:
    docs = load_movies()
    search = MultimodalSearch(docs)
    results = search.search_with_image(image_path)
    return results
