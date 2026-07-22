import string
import os
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

STOPWORDDS = []
with open(STOPWORDS_PATH, "r") as f:
        content = f.read().translate(str.maketrans("", "", string.punctuation)).lower()
        STOPWORDS = content.splitlines()

BM25_K1 = 1.5

BM25_B = 0.75

DEFAULT_QUERY_LIMIT = 5

DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

SCORE_PRECISION = 3

ALPHA = 0.5

K_PARAMETER = 60

OPEN_ROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPEN_ROUTER_FREE_MODEL = "openrouter/free"

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]