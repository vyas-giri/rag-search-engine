import pytest

from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid import HybridRetriever


def rrf(*ranks: int, k: int = 60) -> float:
    return sum(1.0 / (k + r) for r in ranks)


class TestReciprocalRankFusion:
    def test_scores_follow_formula(self):
        fused = reciprocal_rank_fusion({"a": [1, 2], "b": [2, 3]}, k=60)
        scores = {r.doc_id: r.score for r in fused}
        assert scores[1] == pytest.approx(rrf(1))
        assert scores[2] == pytest.approx(rrf(2, 1))
        assert scores[3] == pytest.approx(rrf(2))

    def test_agreement_beats_single_top_rank(self):
        # doc 2 is 2nd and 1st; doc 1 is 1st in only one list
        fused = reciprocal_rank_fusion({"a": [1, 2], "b": [2, 3]})
        assert [r.doc_id for r in fused] == [2, 1, 3]

    def test_union_keeps_docs_found_by_one_retriever(self):
        fused = reciprocal_rank_fusion({"bm25": [10, 11], "dense": [20, 21]})
        assert {r.doc_id for r in fused} == {10, 11, 20, 21}

    def test_ranks_and_sources_recorded(self):
        fused = reciprocal_rank_fusion({"bm25": [1, 2], "dense": [2]})
        by_id = {r.doc_id: r for r in fused}
        assert by_id[2].ranks == {"bm25": 2, "dense": 1}
        assert by_id[2].sources == ["bm25", "dense"]
        assert by_id[1].ranks == {"bm25": 1}
        assert by_id[1].sources == ["bm25"]

    def test_duplicate_within_list_keeps_best_rank(self):
        fused = reciprocal_rank_fusion({"a": [7, 8, 7]})
        assert fused[0].doc_id == 7
        assert fused[0].ranks == {"a": 1}
        assert fused[0].score == pytest.approx(rrf(1))

    def test_ties_are_deterministic(self):
        # 1 and 2 have identical scores (rank 1 + rank 2 each)
        fused = reciprocal_rank_fusion({"a": [1, 2], "b": [2, 1]})
        assert [r.doc_id for r in fused] == [1, 2]

    def test_small_score_differences_are_not_rounded_away(self):
        # rrf(1, 50) vs rrf(2, 45) differ by < 0.001; rounding to 3 dp would tie them
        fused = reciprocal_rank_fusion(
            {"a": [1, 2] + list(range(100, 147)) + [2], "b": [2] + list(range(200, 248)) + [1]}
        )
        assert fused[0].score != pytest.approx(fused[1].score, abs=1e-9)

    def test_limit(self):
        fused = reciprocal_rank_fusion({"a": [1, 2, 3, 4]}, limit=2)
        assert [r.doc_id for r in fused] == [1, 2]

    def test_empty_inputs(self):
        assert reciprocal_rank_fusion({}) == []
        assert reciprocal_rank_fusion({"a": [], "b": []}) == []

    @pytest.mark.parametrize("k", [0, -1])
    def test_invalid_k(self, k):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion({"a": [1]}, k=k)


class FakeRetriever:
    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def retrieve(self, query, k=5):
        self.calls.append((query, k))
        return [{"id": i} for i in self.ids[:k]]


DOCS = [{"id": i, "title": f"Movie {i}", "description": f"Description {i}"} for i in range(1, 6)]


class TestHybridRetriever:
    def make(self, bm25_ids, dense_ids, candidates=10):
        return HybridRetriever(
            documents=DOCS,
            candidates=candidates,
            bm25=FakeRetriever(bm25_ids),
            dense=FakeRetriever(dense_ids),
        )

    def test_returns_full_documents_with_fusion_metadata(self):
        results = self.make([1, 2], [2, 3]).retrieve("q", k=3)
        assert [d["id"] for d in results] == [2, 1, 3]
        top = results[0]
        assert top["title"] == "Movie 2"
        assert top["description"] == "Description 2"
        assert top["bm25Rank"] == 2
        assert top["semanticRank"] == 1
        assert top["sources"] == ["bm25", "dense"]
        assert top["rrfScore"] == pytest.approx(rrf(2, 1))
        assert results[1]["semanticRank"] is None

    def test_requests_candidate_pool_from_each_retriever(self):
        hybrid = self.make([1], [2], candidates=50)
        hybrid.retrieve("bears", k=5)
        assert hybrid.bm25.calls == [("bears", 50)]
        assert hybrid.dense.calls == [("bears", 50)]

    def test_does_not_mutate_source_documents(self):
        docs = [dict(d) for d in DOCS]
        HybridRetriever(documents=docs, bm25=FakeRetriever([1]), dense=FakeRetriever([1])).retrieve("q", 1)
        assert docs == DOCS

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            self.make([1], [1]).retrieve("q", k=0)
