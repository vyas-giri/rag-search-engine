# Agentic RAG Search Engine

Movie discovery over a 5,000-film catalog, built up from keyword search to an
agentic RAG service. Current scope: BM25, dense, hybrid (RRF), cross-encoder
reranking, multimodal image search, and LLM generation via OpenRouter, with a
reproducible retrieval benchmark.

## Status

| Area | Status | Location |
|---|---|---|
| BM25, dense and hybrid (union RRF) retrieval | ✅ Done | `src/retrieval/` |
| Cross-encoder reranking | ✅ Done | `src/retrieval/reranker.py` |
| Retrieval metrics (P/R/MRR/nDCG, latency) + unit tests | ✅ Done | `src/evaluation/`, `tests/` |
| LLM generation, query rewriting, LLM reranking | 🧪 Experimental | `cli/lib/` |
| Multimodal (image → movie) search | 🧪 Experimental | `cli/lib/multimodal_search.py` |
| Qdrant vector store | ⏳ Phase 2 | |
| LangGraph agent (route → retrieve → grade → rewrite → generate) | ⏳ Phase 3 | |
| FastAPI service + Next.js UI | ⏳ Phase 4 | |
| RAGAS answer evaluation | ⏳ Phase 5 | |
| Langfuse tracing, Docker Compose, CI | ⏳ Phase 6 | |

## Retrieval benchmark

10-query golden set, k=5, same machine and cached embeddings.

| Configuration | P@5 | R@5 | MRR@5 | nDCG@5 | Latency |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.420 | 0.541 | 0.783 | 0.667 | ~1,160 ms |
| Dense (all-MiniLM-L6-v2, chunked) | 0.320 | 0.461 | 0.758 | 0.551 | ~440 ms |
| Hybrid, original (intersection RRF) | 0.420 | 0.551 | 0.825 | 0.670 | ~3,940 ms |
| **Hybrid, union RRF** | **0.440** | **0.567** | 0.808 | **0.678** | **~1,660 ms** |
| Hybrid + cross-encoder (5 candidates) | 0.440 | 0.567 | 0.950 | 0.716 | ~2,160 ms |
| Hybrid + cross-encoder (20 candidates) | 0.400 | 0.509 | 0.950 | 0.679 | ~2,200 ms |

What changed from the original hybrid search, and why it got 2.4x faster:

- **Union instead of intersection.** The original scored only documents found
  by *both* retrievers. RRF now scores the union, so keyword-only or
  semantic-only matches survive.
- **Rank maps instead of list scans.** Fusion was O(docs × list length); it is
  now a single pass (`src/retrieval/fusion.py`).
- **Index loaded once**, not unpickled on every query.
- **No rounding before sorting.** Sorting on `round(score, 3)` created ties that
  were broken by movie id instead of relevance.

Caveats: 10 queries is too few for ~0.01 nDCG differences to be significant.
Reranking 20 candidates scores lower than 5 largely because the cross-encoder
surfaces good but unlabeled titles (e.g. *Resident Evil: Apocalypse* for
"zombie apocalypse"), so the golden set needs expanding before choosing a
reranker configuration.

Reproduce:

```bash
uv run python -m src.retrieval.evaluate --retriever hybrid --k 5
uv run python -m src.retrieval.evaluate --retriever hybrid \
  --reranker cross_encoder --k 5 --candidates 20
```

`--retriever` accepts `bm25`, `dense` or `hybrid`.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

**Data.** `data/` and `cache/` are gitignored. Place these files locally:

- `data/movies.json`: the movie catalog
- `data/stopwords.txt`
- `data/golden_dataset.json`: relevance judgments for evaluation

Indexes and embeddings are built into `cache/` on first run, and models are
downloaded on first use. Setting `HF_TOKEN` avoids Hugging Face rate limits.

**LLM features.** Generation, query rewriting and LLM reranking need an
OpenRouter key in `.env` (never committed):

```env
OPENROUTER_API_KEY=your_key_here
```

Retrieval, evaluation and tests run without a key.

## CLI

The original command-line tools in `cli/` still work:

```bash
# Keyword / semantic
uv run python cli/keyword_search_cli.py build
uv run python cli/keyword_search_cli.py bm25search "space adventure" --limit 5
uv run python cli/semantic_search_cli.py search_chunked "revenge drama" --limit 5

# Hybrid, with optional LLM query rewrite and reranking
uv run python cli/hybrid_search_cli.py rrf-search "heist thriller" --limit 5
uv run python cli/hybrid_search_cli.py rrf-search "scary bear movie" \
  --enhance rewrite --rerank-method cross_encoder --limit 5

# RAG generation (needs OPENROUTER_API_KEY)
uv run python cli/augmented_generation_cli.py rag "best dystopian movies"
uv run python cli/augmented_generation_cli.py citations "movies about survival" --limit 5

# Multimodal
uv run python cli/multimodal_search_cli.py image_search data/paddington.jpeg
```

Each script supports `--help`.

## Layout

```text
src/
  retrieval/     # Retriever protocol; BM25, dense, hybrid, fusion, reranker, evaluator
  evaluation/    # Ranking metrics
tests/unit/      # Fusion, metrics and LLM-client tests
cli/             # Original CLI tools; cli/lib holds the underlying implementations
```

`src/` is the application layer that new features go into. Its BM25 and dense
adapters wrap the implementations in `cli/lib`, which will be migrated over
one capability at a time.

## Roadmap

1. **Phase 1 (finish):** validated dataset loading; commit the golden set so
   CI can evaluate; grow it to ~30 queries.
2. **Phase 2, Qdrant:** move dense vectors (one point per chunk) into Qdrant
   behind `src/retrieval/vector_store.py`, then benchmark it, including
   Qdrant-native hybrid search, against the table above.
3. **Phase 3, agent:** a LangGraph workflow (route → retrieve → grade →
   rewrite → generate) with bounded retries, an explicit
   insufficient-information answer, and citations validated against
   retrieved documents.
4. **Phase 4, API and UI:** FastAPI `/chat` returning the answer, sources and
   the graph path taken, plus a Next.js chat UI.
5. **Phase 5, evaluation:** RAGAS faithfulness, answer relevancy, and
   context precision/recall, comparing plain RAG with the agent.
6. **Phase 6, ops:** Langfuse tracing, Docker Compose, and GitHub Actions
   running tests with mocked LLM calls.

## License

GPL-3.0. See [LICENSE](LICENSE).
