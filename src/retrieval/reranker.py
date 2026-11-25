"""Optional reranking adapters."""

from functools import lru_cache
from typing import Literal

from .types import RetrievedDocument

RerankerName = Literal["none", "cross_encoder"]


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    from lib.constants import CROSS_ENCODER_MODEL

    return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank(
    documents: list[RetrievedDocument],
    query: str,
    method: RerankerName = "none",
    k: int | None = None,
) -> list[RetrievedDocument]:
    """Rerank documents, lazily loading the cross-encoder when requested."""
    if method == "none":
        result = list(documents)
    elif method == "cross_encoder":
        pairs = [
            (query, f"{doc.get('title', '')} - {doc.get('description', '')}")
            for doc in documents
        ]
        scores = _cross_encoder().predict(pairs)
        result = [dict(doc, crossEncoderScore=float(score)) for doc, score in zip(documents, scores)]
        result.sort(key=lambda doc: doc["crossEncoderScore"], reverse=True)
    else:
        raise ValueError(f"Unsupported reranker: {method}")
    return result[:k] if k is not None else result
