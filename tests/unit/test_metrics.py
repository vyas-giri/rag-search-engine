import math

import pytest

from src.evaluation.retrieval import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
)

RELEVANT = {"A", "B"}


class TestPrecision:
    def test_basic(self):
        assert precision_at_k(["A", "x", "B", "y"], RELEVANT, 4) == 0.5

    def test_only_top_k_counts(self):
        assert precision_at_k(["x", "A", "B"], RELEVANT, 1) == 0.0

    def test_fewer_results_than_k_still_divides_by_k(self):
        assert precision_at_k(["A"], RELEVANT, 5) == 0.2

    def test_duplicates_count_once(self):
        assert precision_at_k(["A", "A", "A"], RELEVANT, 3) == pytest.approx(1 / 3)

    def test_empty_results(self):
        assert precision_at_k([], RELEVANT, 5) == 0.0


class TestRecall:
    def test_basic(self):
        assert recall_at_k(["A", "x"], RELEVANT, 2) == 0.5

    def test_all_found(self):
        assert recall_at_k(["B", "A"], RELEVANT, 2) == 1.0

    def test_no_relevant_documents(self):
        assert recall_at_k(["A"], set(), 5) == 0.0

    def test_duplicates_count_once(self):
        assert recall_at_k(["A", "A"], RELEVANT, 2) == 0.5


class TestReciprocalRank:
    @pytest.mark.parametrize(
        "retrieved, expected",
        [
            (["A", "x", "y"], 1.0),
            (["x", "A", "y"], 0.5),
            (["x", "y", "A"], 1 / 3),
        ],
    )
    def test_first_hit_position(self, retrieved, expected):
        assert reciprocal_rank(retrieved, RELEVANT, 3) == pytest.approx(expected)

    def test_hit_beyond_k_is_ignored(self):
        assert reciprocal_rank(["x", "y", "A"], RELEVANT, 2) == 0.0

    def test_no_hit(self):
        assert reciprocal_rank(["x", "y"], RELEVANT, 2) == 0.0


class TestNdcg:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["A", "B", "x"], RELEVANT, 3) == pytest.approx(1.0)

    def test_order_matters(self):
        assert ndcg_at_k(["x", "A", "B"], RELEVANT, 3) < ndcg_at_k(["A", "B", "x"], RELEVANT, 3)

    def test_known_value(self):
        dcg = 1 / math.log2(3)  # hit at rank 2
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(["x", "A", "y"], RELEVANT, 3) == pytest.approx(dcg / idcg)

    def test_ideal_capped_at_k(self):
        # 5 relevant docs but k=2: a perfect top-2 is still 1.0
        assert ndcg_at_k(["A", "B"], {"A", "B", "C", "D", "E"}, 2) == pytest.approx(1.0)

    def test_no_relevant_documents(self):
        assert ndcg_at_k(["A", "B"], set(), 2) == 0.0

    def test_duplicates_cannot_exceed_one(self):
        assert ndcg_at_k(["A", "A", "A"], {"A"}, 3) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "fn", [precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k]
)
@pytest.mark.parametrize("k", [0, -3])
def test_invalid_k_raises(fn, k):
    with pytest.raises(ValueError):
        fn(["A"], RELEVANT, k)


def test_retrieval_metrics_schema():
    result = retrieval_metrics(["A", "x"], ["A", "B"], 2)
    assert set(result) == {"precision", "recall", "mrr", "ndcg"}
    assert all(isinstance(v, float) for v in result.values())
