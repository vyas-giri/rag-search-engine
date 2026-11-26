"""Agent graph tests with a scripted fake LLM and retriever (no API key, no Qdrant)."""

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage

from src.agents import prompts
from src.agents.graph import RECURSION_LIMIT, build_graph, run_agent
from src.agents.nodes import AgentNodes, format_documents
from src.agents.schemas import Answer, GradeResult, RewrittenQuery, RouteDecision

DOCS = {
    1: {"id": 1, "title": "Paddington", "description": "A bear from Peru loves marmalade."},
    2: {"id": 2, "title": "Brother Bear", "description": "A boy is turned into a bear."},
    3: {"id": 3, "title": "Jurassic Park", "description": "Dinosaurs escape a theme park."},
}


class FakeLLM:
    """Replays scripted structured outputs per schema and records every call.

    A scripted value may be an exception instance, which is raised instead.
    """

    def __init__(self, responses=None, direct_reply="Hi! I help you find movies."):
        self.responses = {schema: list(items) for schema, items in (responses or {}).items()}
        self.direct_reply = direct_reply
        self.calls = []

    def with_structured_output(self, schema, **_):
        llm = self

        class Bound:
            def invoke(self, messages):
                llm.calls.append((schema.__name__, messages))
                if not llm.responses.get(schema):
                    raise AssertionError(f"unexpected {schema.__name__} call")
                item = llm.responses[schema].pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        return Bound()

    def invoke(self, messages):
        self.calls.append(("direct", messages))
        return AIMessage(self.direct_reply)


class FakeRetriever:
    """Returns the scripted id lists in order, one per call."""

    def __init__(self, *results):
        self.results = list(results)
        self.queries = []

    def retrieve(self, query, k=5):
        self.queries.append(query)
        ids = self.results.pop(0) if self.results else []
        return [dict(DOCS[i]) for i in ids][:k]


def retrieve_route():
    return RouteDecision(route="retrieve", reason="movie request")


def run(llm, retriever, question="bear movies", **kwargs):
    return run_agent(build_graph(llm=llm, retriever=retriever, **kwargs), question)


def test_happy_path_generates_grounded_answer():
    llm = FakeLLM({
        RouteDecision: [retrieve_route()],
        GradeResult: [GradeResult(relevant_ids=[1, 2])],
        Answer: [Answer(answer="Try Paddington.", cited_ids=[1])],
    })
    result = run(llm, FakeRetriever([1, 2, 3]))

    assert result["path"] == ["route", "retrieve", "grade", "generate"]
    assert result["answer"] == "Try Paddington."
    assert [d["id"] for d in result["citations"]] == [1]
    assert result["rewrites"] == 0


def test_direct_route_skips_retrieval():
    retriever = FakeRetriever([1])
    llm = FakeLLM({RouteDecision: [RouteDecision(route="direct", reason="greeting")]})
    result = run(llm, retriever, question="hello")

    assert result["path"] == ["route", "respond_direct"]
    assert result["answer"] == "Hi! I help you find movies."
    assert result["citations"] == []
    assert retriever.queries == []


def test_poor_retrieval_triggers_one_rewrite_then_answers():
    retriever = FakeRetriever([3], [1])
    llm = FakeLLM({
        RouteDecision: [retrieve_route()],
        GradeResult: [GradeResult(relevant_ids=[]), GradeResult(relevant_ids=[1])],
        RewrittenQuery: [RewrittenQuery(query="bear from Peru marmalade")],
        Answer: [Answer(answer="Paddington.", cited_ids=[1])],
    })
    result = run(llm, retriever, question="that marmalade bear")

    assert result["path"] == [
        "route", "retrieve", "grade", "rewrite", "retrieve", "grade", "generate",
    ]
    assert retriever.queries == ["that marmalade bear", "bear from Peru marmalade"]
    assert result["rewrites"] == 1
    assert [d["id"] for d in result["citations"]] == [1]


def test_rewrite_loop_is_bounded():
    retriever = FakeRetriever([3], [3], [3], [3], [3])
    llm = FakeLLM({
        RouteDecision: [retrieve_route()],
        GradeResult: [GradeResult(relevant_ids=[])] * 5,
        RewrittenQuery: [RewrittenQuery(query=f"attempt {i}") for i in range(5)],
    })
    result = run(llm, retriever, max_rewrites=2)

    assert result["path"].count("rewrite") == 2
    assert result["path"][-1] == "insufficient"
    assert result["answer"] == prompts.INSUFFICIENT_ANSWER
    assert len(retriever.queries) == 3


def test_zero_rewrites_goes_straight_to_insufficient():
    llm = FakeLLM({RouteDecision: [retrieve_route()], GradeResult: [GradeResult(relevant_ids=[])]})
    result = run(llm, FakeRetriever([3]), max_rewrites=0)
    assert result["path"] == ["route", "retrieve", "grade", "insufficient"]


def test_grader_cannot_invent_documents():
    llm = FakeLLM({
        RouteDecision: [retrieve_route()],
        GradeResult: [GradeResult(relevant_ids=[1, 999])],
        Answer: [Answer(answer="Paddington.", cited_ids=[1])],
    })
    result = run(llm, FakeRetriever([1, 3]))
    assert [d["id"] for d in result["relevant"]] == [1]


def test_citations_outside_relevant_documents_are_dropped():
    llm = FakeLLM({
        RouteDecision: [retrieve_route()],
        GradeResult: [GradeResult(relevant_ids=[1])],
        # 3 was retrieved but graded irrelevant; 42 was never retrieved.
        Answer: [Answer(answer="Paddington and others.", cited_ids=[1, 3, 42, 1])],
    })
    result = run(llm, FakeRetriever([1, 3]))
    assert [d["id"] for d in result["citations"]] == [1]


def test_answer_without_valid_citations_becomes_insufficient():
    llm = FakeLLM({
        RouteDecision: [retrieve_route()],
        GradeResult: [GradeResult(relevant_ids=[1])],
        Answer: [Answer(answer="Jurassic Park is great.", cited_ids=[3])],
    })
    result = run(llm, FakeRetriever([1, 3]))
    assert result["answer"] == prompts.INSUFFICIENT_ANSWER
    assert result["citations"] == []


def test_malformed_route_output_defaults_to_retrieval():
    llm = FakeLLM({
        RouteDecision: [OutputParserException("bad json")],
        GradeResult: [GradeResult(relevant_ids=[1])],
        Answer: [Answer(answer="Paddington.", cited_ids=[1])],
    })
    result = run(llm, FakeRetriever([1]))
    assert result["path"][:2] == ["route", "retrieve"]


def test_malformed_grade_output_counts_as_nothing_relevant():
    llm = FakeLLM({
        RouteDecision: [retrieve_route()],
        GradeResult: [OutputParserException("bad json")],
    })
    result = run(llm, FakeRetriever([1]), max_rewrites=0)
    assert result["path"][-1] == "insufficient"


def test_empty_retrieval_skips_grader_call():
    llm = FakeLLM({RouteDecision: [retrieve_route()]})
    result = run(llm, FakeRetriever([]), max_rewrites=0)
    assert result["path"] == ["route", "retrieve", "grade", "insufficient"]
    assert [name for name, _ in llm.calls] == ["RouteDecision"]


def test_documents_reach_prompts_as_delimited_data():
    llm = FakeLLM({
        RouteDecision: [retrieve_route()],
        GradeResult: [GradeResult(relevant_ids=[1])],
        Answer: [Answer(answer="Paddington.", cited_ids=[1])],
    })
    run(llm, FakeRetriever([1]))
    grade_messages = next(m for name, m in llm.calls if name == "GradeResult")
    human = grade_messages[-1][1]
    assert "<documents>" in human and "[id=1] Paddington" in human


def test_long_descriptions_are_trimmed():
    doc = {"id": 7, "title": "Epic", "description": "word " * 1000}
    text = format_documents([doc])
    assert len(text) < 1300 and text.endswith("…")


def test_single_node_can_be_invoked_directly():
    graph = build_graph(llm=FakeLLM(), retriever=FakeRetriever([1, 2]))
    update = graph.nodes["retrieve"].invoke({"query": "bears"})
    assert [d["id"] for d in update["documents"]] == [1, 2]
    assert update["path"] == ["retrieve"]


def test_blank_question_rejected():
    with pytest.raises(ValueError):
        run_agent(build_graph(llm=FakeLLM(), retriever=FakeRetriever()), "   ")


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"max_rewrites": -1}])
def test_invalid_node_config(kwargs):
    with pytest.raises(ValueError):
        AgentNodes(FakeLLM(), FakeRetriever(), **kwargs)


def test_recursion_backstop_exceeds_longest_legal_path():
    longest = 1 + 3 * 2 + 3  # route + (retrieve, grade, rewrite) * 2 + retrieve, grade, generate
    assert RECURSION_LIMIT > longest
