# Agentic RAG Search Engine

An incremental AI/ML project for movie discovery over a 5,000-document catalog.
The project currently demonstrates keyword retrieval, dense semantic retrieval,
hybrid RRF search, reranking experiments, multimodal search, and LLM-assisted
generation. The target is a measurable, grounded, observable agentic RAG service.

## Current status

| Area | Status | Current implementation |
|---|---|---|
| BM25 keyword retrieval | Implemented | `cli/lib/keyword_search.py` |
| Dense retrieval | Implemented | `cli/lib/semantic_search.py` |
| Hybrid retrieval with RRF | Implemented | `cli/lib/hybrid_search.py` |
| Cross-encoder reranking | Implemented experimentally | `cli/lib/reranking.py` |
| LLM generation through OpenRouter | Implemented experimentally | `cli/lib/augmented_generation.py` |
| Multimodal image search | Implemented experimentally | `cli/lib/multimodal_search.py` |
| Typed retrieval adapters | Implemented | `src/retrieval/` (union RRF in `src/retrieval/fusion.py`) |
| Retrieval metrics and unit tests | Implemented | `src/evaluation/retrieval.py`, `tests/unit/` |
| Qdrant vector database | Planned | Phase 2 |
| LangGraph agent workflow | Planned | Phase 3 |
| FastAPI service | Planned | Phase 4 |
| RAGAS-style answer evaluation | Planned | Phase 5 |
| Langfuse tracing | Planned | Phase 6 |
| Docker Compose and CI | Planned | Phase 6 |

This status table is intentionally conservative: a skill is marked implemented
only when there is working code in the repository.

## Repository structure

### Current two-layer structure

```text
cli/
  *_cli.py              # Existing command-line entrypoints
  lib/                  # Original retrieval and LLM implementation

src/
  retrieval/            # New stable retrieval interfaces and adapters

data/
  movies.json           # 5,000 movie records
  golden_dataset.json   # Relevance judgments for retrieval evaluation
  paddington.jpeg       # Image-search fixture
  stopwords.txt

cache/                  # Generated indexes and embeddings; not committed
```

The two layers are intentional:

- `cli/` is the current legacy/demo interface. It is still useful for checking
  behavior and is still used by compatibility adapters.
- `src/` is the new application boundary. New features should be added here
  behind typed interfaces and tests.

Do **not** delete `cli/` during Phase 0 or Phase 1. Migrate one capability at a
time, verify behavior parity, and remove old code only after no callers depend
on it.

### Target architecture

```text
src/
  api/
    main.py                 # FastAPI application and HTTP schemas
  agents/
    graph.py                # LangGraph workflow
    state.py                # Typed graph state
    nodes.py                # Route/retrieve/grade/rewrite/generate nodes
    tools.py                # Retrieval and knowledge-graph tools
  retrieval/
    bm25.py                 # Keyword retriever
    dense.py                # Embedding/vector retriever
    hybrid.py               # BM25 + dense fusion
    reranker.py             # Cross-encoder reranking
    vector_store.py         # Qdrant integration
  evaluation/
    retrieval.py            # Precision, recall, MRR, nDCG, latency
    generation.py           # Faithfulness and answer-quality evaluation
    datasets.py              # Dataset loading and validation
  observability/
    tracing.py              # Langfuse spans and structured metadata

tests/
  unit/
  integration/
  evaluation/

data/
docker-compose.yml          # API, Qdrant, and optional observability services
```

The target structure is a destination, not a requirement to create every file
immediately. Each phase adds only the modules needed for its deliverable.

## Phase 0: baseline and architecture

### Baseline dataset

The catalog contains 5,000 movie documents:

```bash
python -c "import json; print(len(json.load(open('data/movies.json'))['movies']))"
```

The relevance judgments are in
[data/golden_dataset.json](data/golden_dataset.json). They contain ten queries
and expected relevant movie titles.

### Legacy baseline command

Run the original evaluation:

```bash
uv run python cli/evaluation_cli.py --limit 5
```

Measured baseline at `k=5`:

| Metric | Value |
|---|---:|
| Mean Precision@5 | 0.420 |
| Mean Recall@5 | 0.551 |
| Mean F1@5 | 0.400 |
| Documents | 5,000 |

The legacy evaluator reports per-query metrics. The mean F1 above is calculated
from those ten per-query F1 values.

Representative legacy queries:

```bash
uv run python cli/hybrid_search_cli.py rrf-search \
  "cute british bear marmalade" --limit 5

uv run python cli/hybrid_search_cli.py rrf-search \
  "dinosaur park" --limit 5

uv run python cli/hybrid_search_cli.py rrf-search \
  "wizards and magic" --limit 5
```

Observed top results include:

- `cute british bear marmalade` → `Paddington` at rank 1
- `dinosaur park` → `Jurassic Park` at rank 1
- `wizards and magic` → `Troll` at rank 1, followed by Harry Potter titles

### New retrieval evaluator

The initial `src/retrieval/` adapters provide a structured evaluation command:

```bash
uv run python -m src.retrieval.evaluate \
  --retriever hybrid \
  --reranker none \
  --k 5
```

Current measured `hybrid + no reranker` baseline:

| Metric | Value |
|---|---:|
| Mean Precision@5 | 0.420 |
| Mean Recall@5 | 0.551 |
| Mean MRR@5 | 0.825 |
| Mean nDCG@5 | 0.670 |
| Mean latency | approximately 3,834 ms/query |

The cross-encoder comparison is:

```bash
uv run python -m src.retrieval.evaluate \
  --retriever hybrid \
  --reranker cross_encoder \
  --k 5
```

Measured cross-encoder result:

| Metric | Value |
|---|---:|
| Mean Precision@5 | 0.420 |
| Mean Recall@5 | 0.551 |
| Mean MRR@5 | 0.950 |
| Mean nDCG@5 | 0.702 |
| Mean latency | approximately 5,876 ms/query |

These are local baselines, not production SLOs. They are useful for comparing
future changes such as Qdrant indexing, candidate-pool sizing, and reranker
optimization.

### Phase 1 foundation fixes: before/after

The original hybrid path in `cli/lib/hybrid_search.py` had four problems:

1. **Intersection instead of union.** Only documents returned by *both* BM25
   and dense retrieval were scored. Standard RRF scores the union, so a strong
   keyword-only or semantic-only match is not dropped.
2. **Quadratic fusion.** For each of 5,000 documents it scanned both ranked
   lists (up to 2,500 entries each) with `next(...)` to find its rank.
3. **Index reloaded per query.** `_bm25_search` unpickled the whole inverted
   index on every query.
4. **Rounded sort key.** Results were sorted on `round(rrfScore, 3)`. RRF
   scores sit between roughly 0.016 and 0.033, so rounding created ties that
   were then broken by movie id instead of relevance.

`src/retrieval/fusion.py` replaces this with a pure `reciprocal_rank_fusion()`
over rank maps: union semantics, unrounded scores, deterministic tie-breaking,
and per-source ranks kept for explainability. `HybridRetriever` takes the
top 100 candidates from each retriever, loads indexes once, and returns full
movie records. The dense adapter previously returned descriptions truncated to
100 characters; it now returns full descriptions, which the reranker and
future generator need.

The evaluator also had a reranking bug: it retrieved only `k` documents before
cross-encoder reranking, so reranking could reorder the top 5 but never
surface anything new. It now retrieves `--candidates` documents first
(default 20 when a reranker is used).

Measured on the 10-query golden set, k=5, same machine, same cached embeddings:

| Configuration | P@5 | R@5 | MRR@5 | nDCG@5 | Latency (ms/query) |
|---|---:|---:|---:|---:|---:|
| BM25 only | 0.420 | 0.541 | 0.783 | 0.667 | ~1,160 |
| Dense only | 0.320 | 0.461 | 0.758 | 0.551 | ~440 |
| Hybrid, before (intersection RRF) | 0.420 | 0.551 | 0.825 | 0.670 | ~3,940 |
| **Hybrid, after (union RRF)** | **0.440** | **0.567** | 0.808 | **0.678** | **~1,660** |
| Hybrid + cross-encoder, before (5 candidates) | 0.420 | 0.551 | 0.950 | 0.702 | ~5,310 |
| Hybrid + cross-encoder, after (5 candidates) | 0.440 | 0.567 | 0.950 | 0.716 | ~2,160 |
| Hybrid + cross-encoder, after (20 candidates) | 0.400 | 0.509 | 0.950 | 0.679 | ~2,200 |

Reproduce with:

```bash
uv run python -m src.retrieval.evaluate --retriever hybrid --k 5
uv run python -m src.retrieval.evaluate --retriever hybrid \
  --reranker cross_encoder --k 5 --candidates 20
```

What the numbers say:

- Union RRF is a small quality gain and a ~2.4x latency reduction. Hybrid
  latency is now roughly BM25 + dense; fusion itself takes milliseconds. The
  remaining cost is the pure-Python BM25 loop over all documents and the
  per-chunk cosine loop in dense search. Phase 2 (Qdrant) targets the latter.
- Reranking 20 candidates scores *lower* than reranking 5. Inspecting the
  results shows the cross-encoder surfacing plausible but unlabeled titles,
  e.g. *Resident Evil: Apocalypse* for "zombie apocalypse" and
  *The Wedding Planner* for "romantic comedy wedding". With only 10 queries and
  incomplete relevance lists, this comparison is not conclusive. Expanding the
  golden set is required before choosing a reranker configuration.
- Ten queries are too few for differences of ~0.01 nDCG to be significant.

### Baseline limitations

- The dataset's relevance lists are incomplete for broad queries, so recall is
  not a perfect measure of user satisfaction.
- The current hybrid implementation uses cached local embeddings and an
  in-process inverted index.
- Cross-encoder reranking improves ranking metrics but is currently expensive.
- There is no answer-level faithfulness or citation evaluation yet.
- There is no API, agent graph, or tracing yet. Unit tests cover fusion,
  metrics, and lazy LLM client creation (`uv run pytest`).
- The golden set has 10 queries with incomplete relevance lists.

## First end-to-end user flow

Use this query as the first vertical slice:

> Find a family-friendly movie involving bears and explain why it matches,
> citing the movie records.

The intended future execution is:

```text
User query
  -> route query
  -> retrieve candidates
  -> grade relevance
  -> rewrite once if retrieval is weak
  -> rerank candidates
  -> generate a grounded answer
  -> validate citations against retrieved documents
  -> return answer, citations, route, latency, and trace metadata
```

Acceptance criteria:

1. At least one relevant movie is retrieved.
2. The answer uses only retrieved movie descriptions.
3. Every citation references a retrieved document.
4. An unsupported claim produces an explicit insufficient-information response.
5. Query retries are bounded; the workflow cannot loop forever.

## Phase 1: retrieval foundation

Phase 1 makes retrieval a clean, testable subsystem. It does not add LangGraph,
FastAPI, Qdrant, Docker, or Langfuse yet.

### Phase 1 outcome

Create one stable retrieval API that can run:

```bash
uv run python -m src.retrieval.evaluate \
  --retriever hybrid \
  --reranker cross_encoder \
  --k 5
```

The command must report per-query and aggregate:

- Precision@k
- Recall@k
- MRR@k
- nDCG@k
- latency

### Phase 1 implementation order

#### Task 1: Define the result contract

Start in [src/retrieval/types.py](src/retrieval/types.py).

Define a consistent result shape for every retriever:

```text
id
title
description
score
sources
ranks
rerank_score
```

Use typed structures rather than passing unstructured dictionaries between
future API and agent code.

#### Task 2: Isolate data loading

Create `src/evaluation/datasets.py` later in this phase, or first add a small
retrieval-local loader if you want to keep the phase narrow.

It should:

- load `data/movies.json`;
- validate that every movie has `id`, `title`, and `description`;
- load `data/golden_dataset.json`;
- fail clearly when a file is missing or malformed.

Do not silently substitute an empty dataset.

#### Task 3: Implement stable retriever adapters

Use these existing files as behavioral references:

- [cli/lib/keyword_search.py](cli/lib/keyword_search.py)
- [cli/lib/semantic_search.py](cli/lib/semantic_search.py)
- [cli/lib/hybrid_search.py](cli/lib/hybrid_search.py)

The new public modules should be:

- `src/retrieval/bm25.py`
- `src/retrieval/dense.py`
- `src/retrieval/hybrid.py`

Each retriever should expose the same method:

```python
retrieve(query: str, k: int) -> list[RetrievedDocument]
```

Preserve the old CLI behavior while adding the new interface.

#### Task 4: Make hybrid retrieval explainable

The hybrid result should expose:

- BM25 rank
- dense rank
- RRF score
- which retrievers returned the document

Use rank maps instead of repeatedly scanning result lists. Preserve document
IDs and do not mutate the source movie dictionaries during scoring.

#### Task 5: Add cross-encoder reranking

Use [cli/lib/reranking.py](cli/lib/reranking.py) as the reference.

The pipeline should be:

```text
hybrid top 20
  -> cross-encoder scores candidates
  -> sort by rerank score
  -> return top k
```

Load the cross-encoder once per process, not once per query. Keep LLM-based
reranking as an experiment; it should not be the default retrieval path.

#### Task 6: Separate evaluation logic from the CLI

The evaluator should eventually be split into:

- `src/evaluation/datasets.py`
- `src/evaluation/retrieval.py`
- a thin command entrypoint in `src/retrieval/evaluate.py` or
  `src/evaluation/__main__.py`

Metric functions should be independently testable.

#### Task 7: Add retrieval tests

Create tests for:

- empty queries and empty result sets;
- Precision@k;
- Recall@k;
- MRR when the first relevant result is at ranks 1, 2, and k;
- nDCG with no relevant documents;
- duplicate titles or duplicate results;
- invalid `k`;
- deterministic result schema;
- legacy CLI compatibility.

Run:

```bash
uv run pytest -q
```

If pytest is not yet a dependency, add it as a development dependency before
writing the tests.

### Phase 1 completion checklist

- [x] `cli/` still works. LLM clients are now created lazily, so importing
  LLM helpers no longer requires `OPENROUTER_API_KEY`.
- [x] `src/retrieval/` exposes typed retriever interfaces.
- [x] BM25, dense, hybrid, and cross-encoder paths share a result contract.
- [x] Evaluation reports Precision, Recall, MRR, nDCG, and latency.
- [x] Metrics have unit tests.
- [x] A retrieval regression report can be reproduced locally.
- [x] No generated cache or secret files are committed.
- [x] README contains the before/after metric table.
- [ ] Dataset loading is validated (Task 2), and `golden_dataset.json` is
  committed so CI can run evaluation. `data/` is currently gitignored.
- [ ] Golden set expanded beyond 10 queries.

## Future phases

### Phase 2: vector store

Add Qdrant behind `src/retrieval/vector_store.py`. Start with local Docker
Qdrant, index movie embeddings, and compare latency and ranking metrics against
the in-process dense baseline.

### Phase 3: agentic RAG

Add `src/agents/` with typed state and a LangGraph workflow:
route → retrieve → grade → rewrite → generate. Add structured retrieval tools
and bounded retries.

### Phase 4: API

Add `src/api/main.py` with `/health`, `/search`, `/chat`, and `/evaluate`.

### Phase 5: evaluation and reliability

Add answer-level evaluation: faithfulness, context precision, context recall,
answer relevancy, citation validity, route accuracy, and bounded failure-mode
tests.

### Phase 6: observability and deployment

Add Langfuse tracing, structured logs, Docker Compose, and GitHub Actions. CI
must use mocked LLM calls and must not require production API keys.

## Local development

Install dependencies:

```bash
uv sync
```

Run an existing CLI:

```bash
uv run python cli/hybrid_search_cli.py \
  rrf-search "talking teddy bear comedy" --limit 5
```

Run the new retrieval evaluator:

```bash
uv run python -m src.retrieval.evaluate \
  --retriever hybrid \
  --reranker none \
  --k 5
```

Run the tests:

```bash
uv run pytest
```

Environment variables for the experimental LLM commands belong in `.env` and
must never be committed. The Hugging Face warning about unauthenticated access
does not invalidate local evaluation; setting `HF_TOKEN` can improve download
limits.