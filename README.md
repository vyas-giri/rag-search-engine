# rag-search-engine

A Python CLI project for experimenting with retrieval-augmented search over a movie dataset, including:

- **Keyword search** (inverted index, TF/IDF, BM25)
- **Semantic search** (SentenceTransformers embeddings)
- **Hybrid search** (BM25 + semantic, weighted and RRF fusion)
- **Query enhancement and reranking** (LLM and cross-encoder based)
- **RAG-style answer generation**
- **Multimodal search** (image-to-movie retrieval with CLIP)

## Repository layout

```text
cli/
  keyword_search_cli.py
  semantic_search_cli.py
  hybrid_search_cli.py
  augmented_generation_cli.py
  multimodal_search_cli.py
  describe_image_cli.py
  evaluation_cli.py
  test_llm.py
  lib/
    constants.py
    keyword_search.py
    semantic_search.py
    hybrid_search.py
    query_enhancement.py
    reranking.py
    augmented_generation.py
    multimodal_search.py
pyproject.toml
```

## Requirements

- Python **3.13+**
- A package manager that can install project dependencies (examples below use `uv`)
- For LLM-powered features: an `OPENROUTER_API_KEY`

## Installation

From the repository root:

```bash
uv sync
```

If you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environment variables

Create a `.env` file in the repository root:

```env
OPENROUTER_API_KEY=your_key_here
```

This is required for:

- `cli/test_llm.py`
- query enhancement (`spell`, `rewrite`, `expand`)
- LLM reranking/evaluation
- augmented generation (`rag`, `summarize`, `citations`, `question`)
- image query rewriting (`describe_image_cli.py`)

## Data and cache directories

The code expects local files under:

- `data/movies.json`
- `data/stopwords.txt`
- `data/golden_dataset.json` (for `evaluation_cli.py`)

and writes generated artifacts to:

- `cache/`

These directories are gitignored, so create/populate them locally before running the CLIs.

## Running commands

Use `python` from the repository root.

### 1) Keyword search

```bash
python cli/keyword_search_cli.py build
python cli/keyword_search_cli.py search "space adventure"
python cli/keyword_search_cli.py tf 42 space
python cli/keyword_search_cli.py idf space
python cli/keyword_search_cli.py tfidf 42 space
python cli/keyword_search_cli.py bm25idf space
python cli/keyword_search_cli.py bm25tf 42 space
python cli/keyword_search_cli.py bm25search "space adventure" --limit 5
```

### 2) Semantic search

```bash
python cli/semantic_search_cli.py verify
python cli/semantic_search_cli.py embed_text "A movie about robots"
python cli/semantic_search_cli.py verify_embeddings
python cli/semantic_search_cli.py embed_query "time travel thriller"
python cli/semantic_search_cli.py search "time travel thriller" --limit 5
python cli/semantic_search_cli.py chunk "long text..." --chunk-size 200 --overlap 0
python cli/semantic_search_cli.py semantic_chunk "long text..." --max-chunk-size 4 --overlap 0
python cli/semantic_search_cli.py embed_chunks
python cli/semantic_search_cli.py search_chunked "revenge drama" --limit 5
```

### 3) Hybrid search

```bash
python cli/hybrid_search_cli.py normalize 1.0 2.0 3.0
python cli/hybrid_search_cli.py weighted-search "heist thriller" --alpha 0.5 --limit 5
python cli/hybrid_search_cli.py rrf-search "heist thriller" --k 60 --limit 5
python cli/hybrid_search_cli.py rrf-search "scary bear movie" --enhance rewrite --rerank-method cross_encoder --limit 5
python cli/hybrid_search_cli.py rrf-search "historical war movie" --evaluate --limit 5
```

### 4) Augmented generation (RAG)

```bash
python cli/augmented_generation_cli.py rag "best dystopian movies"
python cli/augmented_generation_cli.py summarize "animated movies for families" --limit 5
python cli/augmented_generation_cli.py citations "movies about survival" --limit 5
python cli/augmented_generation_cli.py question "Which movies are similar to Interstellar?" --limit 5
```

### 5) Multimodal search

```bash
python cli/multimodal_search_cli.py verify_image_embedding ./path/to/image.jpg
python cli/multimodal_search_cli.py image_search ./path/to/image.jpg
```

### 6) Image-based query rewriting

```bash
python cli/describe_image_cli.py --image ./path/to/image.jpg --query "space movie with a red planet"
```

### 7) OpenRouter connectivity test

```bash
python cli/test_llm.py
```

### 8) Retrieval evaluation

```bash
python cli/evaluation_cli.py --limit 5
```

## Notes

- First runs of embedding/cross-encoder workflows may download models and take longer.
- Some commands depend on generated cache files (indexes/embeddings). If missing, run the corresponding build command first.
- This repository is licensed under **GPL-3.0** (see `LICENSE`).
