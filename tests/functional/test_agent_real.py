"""Live functional tests for the agentic loop (task 12.2).

Run against the active provider's index of data/merged_tickets.json:

    sd-agentic-rag ingest --data-path data/merged_tickets.json --wipe
    uv run pytest -m live tests/functional/test_agent_real.py

LLM output varies, so only deterministic properties are asserted: citations
stay within the grounded allow-list, unanswerable golden queries are refused,
the budget is respected, and on `2x` queries the agent's citation recall is
at least the single-pass baseline's.
"""

import sys
from pathlib import Path

import pytest

from service_checks import skip_if_unavailable

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

GOLDEN = Path(__file__).parent.parent.parent / "data" / "eval" / "agentic_golden.json"


@pytest.fixture(scope="module")
def pipeline():
    try:
        from sd_agentic_rag import SdAgenticRAGPipeline
        pipeline = SdAgenticRAGPipeline()
        if not pipeline.store_enabled:
            pytest.skip("Qdrant not available")
        pipeline.vector_store.client.get_collections()
    except Exception as e:
        skip_if_unavailable(e, "Qdrant")
    from sd_agentic_rag.agent.tools import LookupTool
    from sd_agentic_rag.agent.state import SubGoal
    demo = LookupTool(pipeline.vector_store).run(
        SubGoal(id="g", query="check", tool="lookup", ticket_id="INC980101"), top_k=1)
    assert demo, ("Demo tickets are not indexed for this provider: run "
                  "`sd-agentic-rag ingest --data-path data/merged_tickets.json --wipe`")
    return pipeline


@pytest.fixture(scope="module")
def golden():
    from sd_agentic_rag.sd_agentic_eval import load_golden_set
    return load_golden_set(str(GOLDEN))


@pytest.fixture(scope="module")
def config():
    from sd_agentic_rag.agent import AgentConfig
    return AgentConfig.load().with_overrides(no_memory=True)


@pytest.fixture(scope="module")
def two_x_runs(pipeline, golden, config):
    """Agent and baseline outputs for every `2x` golden query (run once per module)."""
    from sd_agentic_rag.agent import AgenticRAGLoop
    from sd_agentic_rag.sd_agentic_eval import make_baseline_runner
    loop = AgenticRAGLoop(pipeline, config)
    baseline = make_baseline_runner(pipeline)
    runs = []
    for entry in (e for e in golden if e["tag"] == "2x"):
        runs.append((entry, loop.run_turn(entry["query"]), baseline(entry["query"])))
    return runs


@pytest.mark.live
def test_citations_stay_within_the_grounded_allow_list(two_x_runs):
    for entry, turn, _ in two_x_runs:
        assert set(turn.cited_ticket_ids) <= set(turn.grounded_ticket_ids), entry["id"]
        from sd_agentic_rag.sd_agentic_guard import Guardrails
        in_answer = set(Guardrails().extract_citations(turn.answer))
        if not turn.refused:
            assert in_answer <= set(turn.grounded_ticket_ids), (entry["id"], in_answer)


@pytest.mark.live
def test_budget_is_respected(two_x_runs, config):
    budget = config.budget
    for entry, turn, _ in two_x_runs:
        assert turn.loop["iterations"] <= budget.max_iterations, entry["id"]
        assert turn.loop["llm_calls"] <= budget.max_llm_calls + 1, entry["id"]
        assert turn.loop["revisions"] <= budget.max_revisions * budget.max_iterations, entry["id"]


@pytest.mark.live
def test_agent_recall_at_least_baseline_on_2x(two_x_runs):
    from sd_agentic_rag.sd_agentic_eval import citation_recall
    agent = [citation_recall(turn.cited_ticket_ids, e["expected_ticket_ids"]) for e, turn, _ in two_x_runs]
    base = [citation_recall(b.cited_ticket_ids, e["expected_ticket_ids"]) for e, _, b in two_x_runs]
    detail = [(e["id"], a, b) for (e, _, _), a, b in zip(two_x_runs, agent, base)]
    assert sum(agent) / len(agent) >= sum(base) / len(base), detail


@pytest.mark.live
def test_unanswerable_golden_queries_are_refused(pipeline, golden, config):
    from sd_agentic_rag.agent import AgenticRAGLoop
    loop = AgenticRAGLoop(pipeline, config)
    for entry in (e for e in golden if e["tag"] == "unanswerable"):
        turn = loop.run_turn(entry["query"])
        assert turn.refused, (entry["id"], turn.answer)
        assert turn.citations == [] and not turn.partial
