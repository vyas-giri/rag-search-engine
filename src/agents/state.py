"""Shared state for the agent graph."""

import operator
from typing import Annotated, TypedDict

from src.retrieval.types import RetrievedDocument


class AgentState(TypedDict, total=False):
    question: str  # the user's question; never modified
    route: str  # "retrieve" or "direct"
    query: str  # current search query; rewrites replace it
    rewrites: int  # rewrite attempts so far (bounded)
    documents: list[RetrievedDocument]  # latest retrieval results
    relevant: list[RetrievedDocument]  # subset of documents that passed grading
    answer: str
    citations: list[RetrievedDocument]  # validated subset of relevant documents
    # Every node appends its name, giving the path taken through the graph.
    path: Annotated[list[str], operator.add]
