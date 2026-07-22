import os

from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from lib.constants import ALPHA, SCORE_PRECISION, K_PARAMETER, DEFAULT_QUERY_LIMIT, load_movies
from lib.query_enhancement import correct_spelling, enhance_query
from lib.reranking import rerank_results, evaluate_results

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        extendedLimit = limit * 500
        bm25_results = self._bm25_search(query, extendedLimit)
        semantic_results = self.semantic_search.search_chunks(query, extendedLimit)

        bm25_map = {res["id"]: res["bm25_score"] for res in bm25_results}
        semantic_map = {res["id"]: res["score"] for res in semantic_results}

        bm25_scores = list(bm25_map.values())
        semantic_scores = list(semantic_map.values())
        norm_bm25 = normalize_scores(bm25_scores)
        norm_semantic = normalize_scores(semantic_scores)
        
        bm25_norm_map = {doc_id: norm_bm25[i] for i, doc_id in enumerate(bm25_map.keys())}
        semantic_norm_map = {doc_id: norm_semantic[i] for i, doc_id in enumerate(semantic_map.keys())}
        
        ranked_docs = []
        for doc_id, doc in {d["id"]: d for d in self.documents}.items():
            if doc_id in bm25_map and doc_id in semantic_map:
                bm25_score = bm25_map[doc_id]
                semantic_score = semantic_map[doc_id]
                h_score = hybrid_score(bm25_norm_map[doc_id], semantic_norm_map[doc_id], alpha)
                doc_copy = doc.copy()
                doc_copy["bm25Score"] = round(bm25_score, SCORE_PRECISION)
                doc_copy["semanticScore"] = round(semantic_score, SCORE_PRECISION)
                doc_copy["hybridScore"] = round(h_score, SCORE_PRECISION)
                ranked_docs.append(doc_copy)
        
        sorted_docs = sorted(ranked_docs, key=lambda x: x["hybridScore"], reverse=True)
        return sorted_docs[:limit]


    def rrf_search(self, query: str, k: int = K_PARAMETER, limit: int = DEFAULT_QUERY_LIMIT) -> list[dict]:
        extendedLimit = limit * 500
        bm25_results = self._bm25_search(query, extendedLimit)
        semantic_results = self.semantic_search.search_chunks(query, extendedLimit)

        ranked_bm25 = sorted(bm25_results, key=lambda x: x["bm25_score"], reverse=True)
        ranked_semantic = sorted(semantic_results, key=lambda x: x["score"], reverse=True)


        ranked_docs = []
        for doc_id, doc in {d["id"]: d for d in self.documents}.items():
            bm25_rank = next((i for i, res in enumerate(ranked_bm25, start=1) if res["id"] == doc_id), None)
            semantic_rank = next((i for i, res in enumerate(ranked_semantic, start=1) if res["id"] == doc_id), None)
            bm25_rrf_score = rrf_score(bm25_rank, k) if bm25_rank is not None else 0
            semantic_rrf_score = rrf_score(semantic_rank, k) if semantic_rank is not None else 0
            if bm25_rank is not None and semantic_rank is not None:
                rrf_score_value = bm25_rrf_score + semantic_rrf_score
                doc_copy = doc.copy()
                doc_copy["bm25Rank"] = bm25_rank
                doc_copy["semanticRank"] = semantic_rank
                doc_copy["rrfScore"] = round(rrf_score_value, SCORE_PRECISION)
                ranked_docs.append(doc_copy)

        sorted_docs = sorted(ranked_docs, key=lambda x: x["rrfScore"], reverse=True)
        return sorted_docs[:limit]



def normalize_scores(scores: list[float]) -> list[float]:
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    else:
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        return normalized_scores

def normalize_command(scores: list[float]) -> None:
    res = normalize_scores(scores)
    for score in res:
        print(f"{score:.4f}")

def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = ALPHA
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = K_PARAMETER) -> float:
    return 1 / (k + rank)

def weighted_search_command(query: str, alpha: float = ALPHA, limit: int = 5) -> list[dict]:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    res = hybrid_search.weighted_search(query, alpha, limit)
    for i, doc in enumerate(res, start=1):
        print(f"{i}. {doc['title']}")
        print(f"   Hybrid Score: {doc['hybridScore']:.3f}")
        print(f"   BM25: {doc['bm25Score']:.3f}, Semantic: {doc['semanticScore']:.3f}")
        print(f"   {doc['description'][:100]}...\n")

def perform_rrf_search(query: str, k: int = K_PARAMETER, limit: int = DEFAULT_QUERY_LIMIT, enhance: str = None, rerank_method: str = None, evaluate: bool = False) -> list[dict]:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    #print(f"Query: {query}") # debugging

    if enhance is not None:
        query = enhance_query(query, enhance)
        #print(f"Enhanced Query: {query}") # debugging
    
    if rerank_method is not None:
        limit *= 5  # Increase limit for reranking
    
    res = hybrid_search.rrf_search(query, k, limit)
    #print(f"Initial RRF Results (limit={limit}): {[doc['title'] for doc in res]}")  # debugging

    if rerank_method is not None:
        res = rerank_results(res, rerank_method, query, limit // 5)
    
    return res

def rrf_search_command(query: str, k: int = K_PARAMETER, limit: int = DEFAULT_QUERY_LIMIT, enhance: str = None, rerank_method: str = None, evaluate: bool = False) -> None:
    res = perform_rrf_search(query, k, limit, enhance, rerank_method, evaluate)

    for i, doc in enumerate(res, start=1):
        print(f"{i}. {doc['title']}")

        if rerank_method is not None:
            if rerank_method == "individual":
                print(f"   Re-rank Score: {doc['individualRerankScore']:.3f}/10")
            elif rerank_method == "batch":
                print(f"   Re-rank Position: {doc.get('batchRerankPosition', 0)}")
            elif rerank_method == "cross_encoder":
                print(f"   Cross Encoder Score: {doc['crossEncoderScore']:.3f}")

        print(f"   RRF Score: {doc['rrfScore']:.3f}")
        print(f"   BM25 Rank: {doc['bm25Rank']}, Semantic Rank: {doc['semanticRank']}")
        print(f"   {doc['description'][:100]}...\n")

    if evaluate:
        evaluate_results(res, query)
