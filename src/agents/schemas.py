"""Structured outputs the agent asks the LLM for."""

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal["retrieve", "direct"] = Field(
        description="'retrieve' for anything about finding, recommending or describing movies; "
        "'direct' for greetings, questions about the assistant, or off-topic requests"
    )
    reason: str = Field(description="One short sentence explaining the choice")


class GradeResult(BaseModel):
    relevant_ids: list[int] = Field(
        description="ids of the documents that genuinely help answer the question; empty if none"
    )


class RewrittenQuery(BaseModel):
    query: str = Field(description="An improved movie search query, under 12 words")


class Answer(BaseModel):
    answer: str = Field(description="The answer to the user, grounded only in the documents")
    cited_ids: list[int] = Field(description="ids of the documents the answer relies on")
