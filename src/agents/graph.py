"""Build and run the agent graph.

    START → route ─┬─ direct ──────────────────────────→ respond_direct → END
                   └─ retrieve → retrieve → grade ─┬─ relevant ──────→ generate → END
                                    ↑              ├─ none, retries → rewrite ─┐
                                    └──────────────┼───────────────────────────┘
                                                   └─ none, no retries → insufficient → END
"""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.retrieval.types import Retriever

from .nodes import DEFAULT_K, DEFAULT_MAX_REWRITES, AgentNodes
from .state import AgentState

logger = logging.getLogger(__name__)

NODE_NAMES = ("route", "retrieve", "grade", "rewrite", "generate", "respond_direct", "insufficient")
# Longest legal run is route + (retrieve, grade, rewrite) * max_rewrites
# + retrieve, grade, generate; this is a backstop well above that.
RECURSION_LIMIT = 25


def default_retriever(documents: list[dict] | None = None) -> Retriever:
    """Hybrid BM25 + dense RRF, with dense search in Qdrant when it is reachable."""
    from src.retrieval._compat import load_movies
    from src.retrieval.hybrid import HybridRetriever

    documents = documents if documents is not None else load_movies()
    try:
        from src.retrieval.vector_store import QdrantDenseRetriever

        return HybridRetriever(documents, dense=QdrantDenseRetriever(documents))
    except Exception as exc:  # Qdrant down or collection missing
        logger.warning("Qdrant unavailable (%s); using in-process dense search", exc)
        return HybridRetriever(documents)


def build_graph(
    llm: Any = None,
    retriever: Retriever | None = None,
    k: int = DEFAULT_K,
    max_rewrites: int = DEFAULT_MAX_REWRITES,
):
    """Compile the agent graph. Pass ``llm``/``retriever`` to override defaults (e.g. in tests)."""
    if llm is None:
        from .llm import get_chat_model

        llm = get_chat_model()
    if retriever is None:
        retriever = default_retriever()
    nodes = AgentNodes(llm, retriever, k=k, max_rewrites=max_rewrites)

    graph = StateGraph(AgentState)
    for name in NODE_NAMES:
        graph.add_node(name, getattr(nodes, name))

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", nodes.after_route, ["retrieve", "respond_direct"])
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", nodes.after_grade, ["generate", "rewrite", "insufficient"])
    graph.add_edge("rewrite", "retrieve")
    for terminal in ("generate", "respond_direct", "insufficient"):
        graph.add_edge(terminal, END)
    return graph.compile()


def run_agent(graph: Any, question: str, config: dict | None = None) -> AgentState:
    """Invoke a compiled graph on one question with the recursion backstop applied."""
    if not question or not question.strip():
        raise ValueError("question must be non-empty")
    return graph.invoke(
        {"question": question.strip()},
        config={"recursion_limit": RECURSION_LIMIT, **(config or {})},
    )
