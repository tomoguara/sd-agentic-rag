"""Retrieval tools, plan validation and grounding across sub-goals (tasks 2.1-2.4)."""

import pytest

from agent_fakes import (
    DEFAULT_POINTS,
    FakeEmbeddingService,
    FakeRetriever,
    FakeVectorStore,
    make_ticket_point,
)

from sd_agentic_rag.agent.grounding import EvidencePool, ground_evidence, ground_pool
from sd_agentic_rag.agent.state import Evidence, Plan, SubGoal
from sd_agentic_rag.agent.tools import (
    ALLOWED_TOOLS,
    FilteredRetrieveTool,
    LookupTool,
    PlanValidationError,
    RetrieveTool,
    build_tool_registry,
    evidence_from_result,
    render_state_line,
    validate_plan,
)
from sd_agentic_rag.sd_agentic_ground import ContextGrounding
from sd_agentic_rag.sd_agentic_rag_pipeline import SdAgenticRAGPipeline
from sd_agentic_rag.sd_agentic_retrieve import BM25Index


@pytest.fixture
def store():
    return FakeVectorStore()


@pytest.fixture
def retriever(store):
    return FakeRetriever(store, FakeEmbeddingService())


def goal(tool="retrieve", query="slow API latency", **kwargs):
    return SubGoal(id=kwargs.pop("id", "g1"), query=query, tool=tool, **kwargs)


# -- 2.1 shared retrieve-then-rank helper ------------------------------------

class FakeRerank:
    def run(self, query, documents, top_k=None, enabled=None):
        ranked = sorted(documents, key=lambda d: d["id"], reverse=True)[:top_k]
        return [dict(d, rerank_score=0.9 - i * 0.1) for i, d in enumerate(ranked)]


def _bare_pipeline(retriever, rerank=None):
    pipeline = object.__new__(SdAgenticRAGPipeline)
    pipeline.store_enabled = True
    pipeline.hybrid_retriever = retriever
    pipeline.rerank_pipeline = rerank
    pipeline.rerank_enabled = rerank is not None
    pipeline.ground_enabled = True
    pipeline.ground_pipeline = ContextGrounding(max_passages=5, max_tokens=4000, min_relevance_score=0.05)
    return pipeline


@pytest.mark.parametrize("rerank", [None, FakeRerank()])
def test_answer_path_and_retrieve_tool_yield_identical_results(retriever, rerank):
    pipeline = _bare_pipeline(retriever, rerank)
    base = pipeline.hybrid_search_with_ground("slow API latency", top_k=2)["reranked_results"]
    tool = RetrieveTool(retriever, rerank, rerank_enabled=rerank is not None)
    evidence = tool.run(goal(), top_k=2)
    assert [e.point_id for e in evidence] == [str(r["id"]) for r in base]
    assert [e.ticket_id for e in evidence] == [r["payload"]["ticket_id"] for r in base]
    for e, r in zip(evidence, base):
        for key in ("rerank_score", "fused_score"):
            assert e.scores.get(key) == r.get(key)
    # Both asked the retriever for the same candidate set
    assert retriever.calls[0]["top_k"] == retriever.calls[1]["top_k"] == 100


# -- 2.2 tools -----------------------------------------------------------

def test_state_line_renders_present_fields_and_omits_missing():
    assert render_state_line({"status": "Resolved", "priority": "P2", "severity": "High",
                              "sla_minutes": 60, "assigned_to": "Support Team"}) == \
        "Status: Resolved | Priority: P2 | Severity: High | SLA: 60 min | Assigned: Support Team"
    assert render_state_line({"status": "Open", "sla_minutes": None, "assigned_to": ""}) == "Status: Open"
    assert render_state_line({}) == ""


def test_evidence_without_ticket_id_is_discarded():
    assert evidence_from_result({"id": "x", "payload": {"chunk_text": "orphan"}}, "retrieve") is None


def test_retrieve_tool_returns_state_prefixed_ticket_attributed_evidence(retriever):
    evidence = RetrieveTool(retriever).run(goal(), top_k=5)
    assert evidence, "expected matches"
    top = evidence[0]
    assert top.ticket_id == "LAT001234" and top.tool == "retrieve"
    assert top.text.startswith("Status: Resolved | Priority: P1 | Severity: High | SLA: 240 min")
    assert top.metadata["service"] == "Database Services"
    assert "chunk_text" not in top.metadata


def test_bm25_only_match_carries_real_ticket_id(store):
    bm25 = BM25Index()
    bm25.build([p["payload"]["chunk_text"] for p in store.points], [p["id"] for p in store.points],
               [p["payload"] for p in store.points])
    # No vector matches for this query: everything comes from BM25
    retriever = FakeRetriever(store, FakeEmbeddingService(vectors={"certificate": [0.0, 0.0, 0.0]}), bm25)
    evidence = RetrieveTool(retriever).run(goal(query="certificate"), top_k=5)
    assert [e.ticket_id for e in evidence] == ["NET000111"]
    assert evidence[0].retriever == "bm25" and evidence[0].metadata["status"] == "Open"


def test_lookup_returns_all_chunks_in_order(store):
    evidence = LookupTool(store).run(goal("lookup", ticket_id="LAT001234"), top_k=5)
    assert [e.point_id for e in evidence] == ["p1", "p4"]
    assert all(e.ticket_id == "LAT001234" and e.tool == "lookup" for e in evidence)
    assert "Status: Resolved" in evidence[0].text and "SLA: 240 min" in evidence[0].text


def test_lookup_unknown_id_returns_no_evidence(store):
    assert LookupTool(store).run(goal("lookup", ticket_id="ZZZ000000"), top_k=5) == []


@pytest.mark.parametrize("field, value", [
    ("service", "Database Services"),
    ("severity", "Low"),
    ("ticket_type", "Incident"),
    ("status", "Resolved"),
])
def test_filtered_retrieve_filters_both_retrievers(store, field, value):
    bm25 = BM25Index()
    bm25.build([p["payload"]["chunk_text"] for p in store.points], [p["id"] for p in store.points],
               [p["payload"] for p in store.points])
    embed = FakeEmbeddingService(vectors={"fixed": [1.0, 1.0, 1.0]})
    retriever = FakeRetriever(store, embed, bm25)
    evidence = FilteredRetrieveTool(retriever).run(goal("filtered_retrieve", query="fixed", filters={field: value}),
                                                   top_k=10)
    expected = {p["id"] for p in DEFAULT_POINTS if p["payload"].get(field) == value}
    assert {e.point_id for e in evidence} == expected
    assert all(e.metadata[field] == value for e in evidence)
    assert store.searches[-1]["filter_conditions"][0].key == field


def test_registry_has_exactly_the_three_tools(store, retriever):
    class P:
        pass
    pipeline = P()
    pipeline.hybrid_retriever, pipeline.rerank_pipeline, pipeline.vector_store = retriever, None, store
    assert tuple(build_tool_registry(pipeline)) == ALLOWED_TOOLS == ("retrieve", "lookup", "filtered_retrieve")


# -- 2.3 plan validation -------------------------------------------------

def test_plan_with_unknown_tool_is_rejected():
    with pytest.raises(PlanValidationError, match="web_search"):
        validate_plan(Plan(sub_goals=[goal("web_search")]))


def test_filter_on_assigned_to_is_rejected():
    with pytest.raises(PlanValidationError, match="assigned_to"):
        validate_plan(Plan(sub_goals=[goal("filtered_retrieve", filters={"assigned_to": "Support Team"})]))


@pytest.mark.parametrize("bad", [
    goal("lookup", ticket_id=None),
    goal("lookup", ticket_id="not-an-id"),
    goal("filtered_retrieve", filters={}),
    goal(query="  "),
])
def test_malformed_sub_goals_are_rejected(bad):
    with pytest.raises(PlanValidationError):
        validate_plan(Plan(sub_goals=[bad]))


def test_valid_plan_and_limits():
    plan = Plan(sub_goals=[goal(id="g1"), goal("lookup", id="g2", ticket_id="LAT001234"),
                           goal("filtered_retrieve", id="g3", filters={"status": "Resolved"})])
    assert validate_plan(plan, max_sub_goals=3) is plan
    with pytest.raises(PlanValidationError, match="max 2"):
        validate_plan(plan, max_sub_goals=2)
    with pytest.raises(PlanValidationError, match="unique"):
        validate_plan(Plan(sub_goals=[goal(id="g1"), goal(id="g1")]))


# -- 2.4 grounding across sub-goals -----------------------------------------

def _evidence(point_id, ticket_id, score, text=None):
    return Evidence(point_id=point_id, ticket_id=ticket_id, text=text or f"passage {point_id}",
                    metadata={}, tool="retrieve", score=score, scores={"fused_score": score})


def test_single_sub_goal_grounds_exactly_like_ground_context():
    grounder = ContextGrounding(max_passages=3, max_tokens=4000, min_relevance_score=0.05)
    items = [_evidence(f"p{i}", f"T{i:06d}", 0.1 * i) for i in range(1, 6)]
    pool = EvidencePool()
    pool.add("g1", items)
    expected = grounder.ground_context(reranked_results=[e.to_ground_result() for e in items], query="q")
    assert ground_pool(grounder, pool, {"g1": "q"}) == expected


def test_two_sub_goals_each_keep_their_passages():
    grounder = ContextGrounding(max_passages=2, max_tokens=4000, min_relevance_score=0.05)
    pool = EvidencePool()
    # g1's scores are far higher; a single ground_context would keep only g1's passages
    pool.add("g1", [_evidence("a1", "AAA000001", 10.0), _evidence("a2", "AAA000002", 9.0),
                    _evidence("a3", "AAA000003", 8.0)])
    pool.add("g2", [_evidence("b1", "BBB000001", 0.01)])
    grounded = ground_pool(grounder, pool, {"g1": "status", "g2": "sla"})
    ids = [c["ticket_id"] for c in grounded.citations]
    assert ids == ["AAA000001", "AAA000002", "BBB000001"]
    assert [c["rank"] for c in grounded.citations] == [1, 2, 3]
    assert "[CITATION:BBB000001]" in grounded.context_text


def test_merge_drops_a_passage_shared_by_two_sub_goals():
    grounder = ContextGrounding(max_passages=5, max_tokens=4000, min_relevance_score=0.05)
    shared = _evidence("s1", "LAT001234", 1.0)
    pool = EvidencePool()
    pool.add("g1", [shared])
    pool.add("g2", [shared, _evidence("x1", "NET000111", 0.5)])
    grounded = ground_pool(grounder, pool, {})
    assert [c["ticket_id"] for c in grounded.citations] == ["LAT001234", "NET000111"]
    assert grounded.context_text.count("[CITATION:LAT001234]") == 1


def test_merged_context_is_truncated_to_the_token_budget():
    grounder = ContextGrounding(max_passages=5, max_tokens=60, min_relevance_score=0.05)
    pool = EvidencePool()
    pool.add("g1", [_evidence("a", "AAA000001", 1.0, "x" * 150)])
    pool.add("g2", [_evidence("b", "BBB000001", 0.5, "y" * 150)])
    grounded = ground_pool(grounder, pool, {})
    assert grounded.total_tokens <= 60
    assert [c["ticket_id"] for c in grounded.citations] == ["AAA000001"]


def test_empty_pool_grounds_nothing():
    grounded = ground_pool(ContextGrounding(max_passages=5, max_tokens=4000), EvidencePool(), {})
    assert grounded.citations == [] and grounded.context_text == ""
