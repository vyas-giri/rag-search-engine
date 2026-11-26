"""Graph nodes. Each takes the state and returns a partial update.

Nodes receive the LLM and retriever through ``AgentNodes`` so tests can pass
fakes, with no API key and no Qdrant.
"""

import logging
from typing import Any, Literal

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError

from src.retrieval.types import RetrievedDocument, Retriever

from . import prompts
from .schemas import Answer, GradeResult, RewrittenQuery, RouteDecision
from .state import AgentState

logger = logging.getLogger(__name__)

DEFAULT_K = 5
DEFAULT_MAX_REWRITES = 2
# Long plot descriptions are trimmed in prompts to bound token cost.
MAX_DESCRIPTION_CHARS = 1200


def format_documents(documents: list[RetrievedDocument]) -> str:
    blocks = []
    for doc in documents:
        description = doc.get("description", "")
        if len(description) > MAX_DESCRIPTION_CHARS:
            description = description[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0] + " …"
        blocks.append(f"[id={doc['id']}] {doc['title']}\n{description}")
    return "\n\n".join(blocks)


class AgentNodes:
    def __init__(
        self,
        llm: Any,
        retriever: Retriever,
        k: int = DEFAULT_K,
        max_rewrites: int = DEFAULT_MAX_REWRITES,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if max_rewrites < 0:
            raise ValueError(f"max_rewrites must be non-negative, got {max_rewrites}")
        self.llm = llm
        self.retriever = retriever
        self.k = k
        self.max_rewrites = max_rewrites

    def _structured(self, schema: type[BaseModel], system: str, human: str) -> BaseModel | None:
        """Ask for a structured object; None if the model's output is malformed.

        Network and auth errors are not caught: those should fail loudly.
        """
        try:
            return self.llm.with_structured_output(schema).invoke(
                [("system", system), ("human", human)]
            )
        except (OutputParserException, ValidationError) as exc:
            logger.warning("Malformed %s from LLM: %s", schema.__name__, exc)
            return None

    # --- nodes -------------------------------------------------------------

    def route(self, state: AgentState) -> dict:
        decision = self._structured(RouteDecision, prompts.ROUTE_SYSTEM, state["question"])
        # When unsure, searching is the safer default for a search assistant.
        route = decision.route if decision else "retrieve"
        return {"route": route, "query": state["question"], "rewrites": 0, "path": ["route"]}

    def retrieve(self, state: AgentState) -> dict:
        documents = self.retriever.retrieve(state["query"], self.k)
        return {"documents": documents, "path": ["retrieve"]}

    def grade(self, state: AgentState) -> dict:
        documents = state.get("documents", [])
        if not documents:
            return {"relevant": [], "path": ["grade"]}
        result = self._structured(
            GradeResult,
            prompts.GRADE_SYSTEM,
            prompts.GRADE_HUMAN.format(
                question=state["question"], documents=format_documents(documents)
            ),
        )
        # Keep only ids that were actually retrieved: models can invent ids.
        keep = set(result.relevant_ids) if result else set()
        relevant = [doc for doc in documents if doc["id"] in keep]
        return {"relevant": relevant, "path": ["grade"]}

    def rewrite(self, state: AgentState) -> dict:
        result = self._structured(
            RewrittenQuery,
            prompts.REWRITE_SYSTEM,
            prompts.REWRITE_HUMAN.format(question=state["question"], query=state["query"]),
        )
        query = result.query.strip() if result and result.query.strip() else state["query"]
        return {"query": query, "rewrites": state.get("rewrites", 0) + 1, "path": ["rewrite"]}

    def generate(self, state: AgentState) -> dict:
        relevant = state.get("relevant", [])
        result = self._structured(
            Answer,
            prompts.GENERATE_SYSTEM,
            prompts.GENERATE_HUMAN.format(
                question=state["question"], documents=format_documents(relevant)
            ),
        )
        by_id = {doc["id"]: doc for doc in relevant}
        cited_ids = list(dict.fromkeys(result.cited_ids)) if result else []
        citations = [by_id[i] for i in cited_ids if i in by_id]
        if result is None or not citations:
            # An answer that cites no retrieved movie is not grounded.
            return {
                "answer": prompts.INSUFFICIENT_ANSWER,
                "citations": [],
                "path": ["generate"],
            }
        return {"answer": result.answer, "citations": citations, "path": ["generate"]}

    def respond_direct(self, state: AgentState) -> dict:
        message = self.llm.invoke([("system", prompts.DIRECT_SYSTEM), ("human", state["question"])])
        return {"answer": message.content, "citations": [], "path": ["respond_direct"]}

    def insufficient(self, state: AgentState) -> dict:
        return {"answer": prompts.INSUFFICIENT_ANSWER, "citations": [], "path": ["insufficient"]}

    # --- routing functions (conditional edges, not nodes) -------------------

    @staticmethod
    def after_route(state: AgentState) -> Literal["retrieve", "respond_direct"]:
        return "retrieve" if state["route"] == "retrieve" else "respond_direct"

    def after_grade(self, state: AgentState) -> Literal["generate", "rewrite", "insufficient"]:
        if state.get("relevant"):
            return "generate"
        if state.get("rewrites", 0) < self.max_rewrites:
            return "rewrite"
        return "insufficient"
