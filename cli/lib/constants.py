import string
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")

STOPWORDDS = []
with open(STOPWORDS_PATH, "r") as f:
        content = f.read().translate(str.maketrans("", "", string.punctuation)).lower()
        STOPWORDS = content.splitlines()

BM25_K1 = 1.5

BM25_B = 0.75

DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

SCORE_PRECISION = 3
