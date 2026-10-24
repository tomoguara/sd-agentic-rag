"""The agentic loop: iterations, revisions, exhaustion, bounded autonomy, 1x equivalence (tasks 7.1-7.4)."""

import pytest

from agent_fakes import FakePipeline, ScriptedChatClient, llm_from

from sd_agentic_rag.agent.config import AgentConfig
from sd_agentic_rag.agent.loop import AgenticRAGLoop
from sd_agentic_rag.agent.prompts import CRITIC_SYSTEM
from sd_agentic_rag.sd_agentic_guard import STANDARD_REFUSAL
from sd_agentic_rag.sd_agentic_rag_pipeline import SdAgenticRAGPipeline


SUFFICIENT = {"sufficient": True, "covered": ["g1"], "missing": []}
INSUFFICIENT = {"sufficient": False, "covered": [], "missing": ["g1"], "refine_hint": ""}
CITED = "The latency was fixed by adding a composite index. [CITATION:LAT001234]"


def make_config(planner="llm", critic="llm", self_review="heuristic", **budget):
    config = AgentConfig()
    config.planner.mode = planner
    config.critic.mode = critic
    config.self_review.mode = self_review
    config.memory.enabled = False
    for key, value in budget.items():
        setattr(config.budget, key, value)
    config.validate()
    return config


def make_loop(responses, answers=None, config=None, clock=None, client=None, **pipeline_kwargs):
    pipeline = FakePipeline(answers=answers or [CITED], **pipeline_kwargs)
    client = client or ScriptedChatClient(responses)
    kwargs = {"clock": clock} if clock else {}
    loop = AgenticRAGLoop(pipeline, config or make_config(), llm_call=llm_from(client), **kwargs)
    return loop, pipeline, client


def events(result, name):
    return [e for e in result.trace if e["event"] == name]


# -- 7.1 run_turn -------------------------------------------------------------

def test_one_iteration_pass():
    loop, pipeline, client = make_loop([
        {"sub_goals": [{"id": "g1", "query": "slow API latency", "tool": "retrieve"}]},
        SUFFICIENT,
    ])
    result = loop.run_turn("Why is the API slow?")
    assert result.answer == CITED and not result.partial and not result.refused
    assert result.stop_reason == "answered"
    assert result.loop["iterations"] == 1
    assert result.loop["llm_calls"] == 3              # plan, critic, generate
    assert result.cited_ticket_ids == ["LAT001234"]
    assert result.citations[0]["tool"] == "retrieve"
    assert result.guard["allowed"] is True


def test_two_iterations_pool_evidence():
    loop, pipeline, client = make_loop([
        {"sub_goals": [
            {"id": "g1", "query": "LAT001234 status", "tool": "lookup", "ticket_id": "LAT001234"},
            {"id": "g2", "query": "SLA terms", "tool": "retrieve"},     # finds nothing
        ]},
        {"sufficient": False, "covered": ["g1"], "missing": ["g2"], "refine_hint": "look for replication"},
        {"sub_goals": [{"id": "g2", "query": "DB01 replication lag", "tool": "retrieve"}]},
        {"sufficient": True, "covered": ["g1", "g2"], "missing": []},
    ], answers=["LAT001234 is resolved [CITATION:LAT001234]. Replication lag was fixed by tuning WAL. "
                "[CITATION:LAT002345]"])
    result = loop.run_turn("Is LAT001234 resolved and how was replication lag fixed?")
    first, second = result.iterations
    assert [c["sub_goal"] for c in first.tool_calls] == ["g1", "g2"]
    assert first.tool_calls[1]["count"] == 0
    assert [c["sub_goal"] for c in second.tool_calls] == ["g2"]       # only the missing sub-goal re-runs
    # Iteration 2 grounds iteration 1's lookup evidence together with the new evidence
    assert "LAT001234" in second.grounded_ticket_ids and "LAT002345" in second.grounded_ticket_ids
    assert result.cited_ticket_ids == ["LAT001234", "LAT002345"]
    assert result.loop["iterations"] == 2 and not result.partial


def test_self_review_revision_succeeds():
    uncited = ("The latency was fixed by adding a composite index on the main query path. "
               "The connection pool was also doubled and caching was added.")
    loop, pipeline, client = make_loop(
        [{"sub_goals": [{"id": "g1", "query": "slow API latency", "tool": "retrieve"}]}, SUFFICIENT],
        answers=[uncited, CITED],
    )
    result = loop.run_turn("Why is the API slow?")
    assert result.answer == CITED
    drafts = result.iterations[0].drafts
    assert [d["self_review"]["passed"] for d in drafts] == [False, True]
    revision_prompt = pipeline.generate_pipeline.calls[1]["system_prompt"]
    assert "previous draft was rejected" in revision_prompt and "citation" in revision_prompt.lower()
    assert result.loop["revisions"] == 1


def test_failed_revision_returns_control_to_the_planner():
    uncited = "The latency was fixed by adding a composite index on the main query path today."
    loop, pipeline, client = make_loop(
        [SUFFICIENT, SUFFICIENT],
        answers=[uncited, uncited, CITED],
        config=make_config(planner="heuristic"),
    )
    result = loop.run_turn("Why is the API slow?")
    assert result.loop["iterations"] == 2
    assert len(result.iterations[0].drafts) == 2          # draft + one revision
    assert result.answer == CITED and not result.partial
    assert events(result, "refine")


def test_refine_that_repeats_an_executed_call_is_not_run_again():
    lookup = {"sub_goals": [{"id": "g1", "query": "ZZZ999999", "tool": "lookup", "ticket_id": "ZZZ999999"}]}
    loop, pipeline, client = make_loop([lookup, lookup, lookup])
    result = loop.run_turn("What is the status of ticket ZZZ999999?")
    calls = [c for r in result.iterations for c in r.tool_calls]
    assert [c["tool"] for c in calls].count("lookup") == 1    # the unknown ID is not looked up again
    assert calls[1]["tool"] == "retrieve" and calls[1]["query"] == "ZZZ999999"   # broadened instead (D18)
    assert result.refused
    assert events(result, "refine_repeated")


def test_reworded_lookup_counts_as_a_repeat():
    first = {"sub_goals": [{"id": "g1", "query": "status of INC follow-up", "tool": "lookup", "ticket_id": "LAT001234"},
                           {"id": "g2", "query": "follow-up work", "tool": "lookup", "ticket_id": "LAT001234"}]}
    reworded = {"sub_goals": [{"id": "g2", "query": "LAT001234", "tool": "lookup", "ticket_id": "lat001234"}]}
    loop, pipeline, client = make_loop([first, INSUFFICIENT | {"missing": ["g2"]}, reworded, SUFFICIENT])
    result = loop.run_turn("What follow-up work came out of LAT001234?")
    assert events(result, "refine_repeated")
    # The repeat falls through to the heuristic broadening: a retrieve of the ID plus the query
    second = result.iterations[1].tool_calls
    assert [(c["sub_goal"], c["tool"], c["query"]) for c in second] == [("g2", "retrieve", "LAT001234 follow-up work")]


def test_missing_lookup_finds_a_ticket_that_references_it():
    from agent_fakes import DEFAULT_POINTS, make_ticket_point
    from sd_agentic_rag.sd_agentic_retrieve import BM25Index
    points = DEFAULT_POINTS + [make_ticket_point(
        "p9", "CHG000999", "Follow-up to LAT001234: add slow query alerting.", vector=[0.0, 0.0, 0.0],
        status="In Progress", sla_minutes=1440)]
    loop, pipeline, client = make_loop(
        [{"sufficient": False, "covered": ["g1"], "missing": ["g2"]},
         {"sufficient": True, "covered": ["g1", "g2"], "missing": []}],
        answers=["LAT001234 is resolved [CITATION:LAT001234]. Its follow-up CHG000999 is in progress. "
                 "[CITATION:CHG000999]"],
        config=make_config(planner="heuristic"), points=points)
    bm25 = BM25Index()
    bm25.build([p["payload"]["chunk_text"] for p in points], [p["id"] for p in points], [p["payload"] for p in points])
    pipeline.hybrid_retriever.bm25_index = bm25
    result = loop.run_turn("Is LAT001234 resolved, and what follow-up came out of LAT001234?")
    second = result.iterations[1].tool_calls
    assert second[0]["tool"] == "retrieve" and "LAT001234" in second[0]["query"]
    assert "CHG000999" in second[0]["evidence_ticket_ids"]
    assert result.cited_ticket_ids == ["LAT001234", "CHG000999"]


def test_repeated_retrieve_falls_back_to_heuristic_broadening():
    plan = {"sub_goals": [{"id": "g1", "query": "coffee machine", "tool": "retrieve"}]}
    loop, pipeline, client = make_loop([plan, plan, plan, plan])
    result = loop.run_turn("How do I configure the coffee machine?")
    top_ks = [c for r in result.iterations for c in r.tool_calls]
    assert len(top_ks) == 3
    assert [r.plan["sub_goals"][0]["top_k"] for r in result.iterations] == [5, 10, 20]


# -- 7.2 exhaustion -------------------------------------------------------------

def test_critic_never_satisfied_stops_at_max_iterations_with_partial_answer():
    loop, pipeline, client = make_loop([INSUFFICIENT] * 10, config=make_config(planner="heuristic"))
    result = loop.run_turn("Why is the API slow?")
    assert result.loop["iterations"] == 3
    assert result.stop_reason == "max_iterations"
    assert result.partial is True and result.answer == CITED
    assert result.loop["reserved_call_used"] is True
    assert events(result, "exhaust")


def test_token_limit():
    loop, pipeline, client = make_loop([INSUFFICIENT] * 10, client=ScriptedChatClient([INSUFFICIENT] * 10, tokens=50),
                                       config=make_config(planner="heuristic", max_total_tokens=60))
    result = loop.run_turn("Why is the API slow?")
    assert result.stop_reason == "max_total_tokens"
    assert result.loop["iterations"] == 2
    assert result.partial is True


class TickingClient(ScriptedChatClient):
    """Each LLM call takes `seconds` on the fake clock."""

    def __init__(self, responses, clock, seconds):
        super().__init__(responses)
        self.clock, self.seconds = clock, seconds

    def _create(self, **kwargs):
        self.clock.now += self.seconds
        return super()._create(**kwargs)


class FakeClock:
    now = 0.0

    def __call__(self):
        return self.now


def test_wall_clock_limit():
    clock = FakeClock()
    client = TickingClient([INSUFFICIENT] * 10, clock, seconds=40)
    loop, pipeline, _ = make_loop([], client=client, clock=clock,
                                  config=make_config(planner="heuristic", max_wall_seconds=60))
    result = loop.run_turn("Why is the API slow?")
    assert result.stop_reason == "max_wall_seconds"
    assert result.loop["iterations"] == 2
    assert result.partial is True      # the reserved call still answers from grounded evidence


def test_exhaustion_with_no_evidence_is_the_standard_refusal():
    loop, pipeline, client = make_loop([], config=make_config(planner="heuristic"))
    result = loop.run_turn("How do I configure the coffee machine?")    # matches no ticket
    assert result.answer == STANDARD_REFUSAL
    assert result.refused and not result.partial and result.citations == []
    assert pipeline.generate_pipeline.calls == []
    assert client.requests == []        # zero-evidence Critic verdicts need no LLM


def test_partial_draft_failing_the_gates_is_refused():
    loop, pipeline, client = make_loop([INSUFFICIENT] * 10, config=make_config(planner="heuristic"),
                                       answers=["The index fixed it. [CITATION:FAKE000001]"])
    result = loop.run_turn("Why is the API slow?")
    assert result.answer == STANDARD_REFUSAL and not result.partial


# -- 7.3 bounded autonomy ---------------------------------------------------------

def test_llm_plan_with_unknown_tool_is_rejected():
    loop, pipeline, client = make_loop([
        {"sub_goals": [{"id": "g1", "query": "search the web", "tool": "web_search"}]},
        SUFFICIENT,
    ])
    result = loop.run_turn("Why is the API slow?")
    assert events(result, "planner_fallback")
    assert [c["tool"] for c in result.iterations[0].tool_calls] == ["retrieve"]
    assert result.loop["fallbacks"] == ["planner_fallback"]


def test_critic_cannot_add_citable_evidence():
    loop, pipeline, client = make_loop(
        [{"sufficient": True, "covered": ["g1", "FAKE000001"], "missing": [],
          "evidence": [{"ticket_id": "FAKE000001", "text": "made up"}]}] * 5,
        answers=["Per the judge, the fix was a reboot. [CITATION:FAKE000001]"],
        config=make_config(planner="heuristic"),
    )
    result = loop.run_turn("Why is the API slow?")
    assert "FAKE000001" not in result.grounded_ticket_ids
    assert all("FAKE000001" not in r.grounded_ticket_ids for r in result.iterations)
    assert "FAKE000001" not in result.answer and result.citations == []


def test_llm_self_review_cannot_pass_a_failing_draft():
    loop, pipeline, client = make_loop(
        [SUFFICIENT] * 5 + [{"unsupported_claims": []}] * 5,
        answers=["The fix was a reboot. [CITATION:FAKE000001]"],
        config=make_config(planner="heuristic", self_review="llm"),
    )
    result = loop.run_turn("Why is the API slow?")
    assert "FAKE000001" not in result.answer
    reviews = [d["self_review"] for r in result.iterations for d in r.drafts]
    assert reviews and not any(r["passed"] for r in reviews)
    # The tighten-only LLM check never ran for these failing drafts
    assert not any("unsupported" in m["content"] for req in client.requests for m in req["messages"]
                   if m["role"] == "system")


@pytest.mark.parametrize("max_llm_calls", [1, 2, 3, 5])
def test_llm_calls_never_exceed_max_plus_one(max_llm_calls):
    uncited = "The latency was fixed by adding a composite index on the main query path today."
    client = ScriptedChatClient([{"sub_goals": [{"id": "g1", "query": "slow API latency", "tool": "retrieve"}]}]
                                + [INSUFFICIENT, SUFFICIENT] * 10)
    loop, pipeline, _ = make_loop([], client=client, answers=[uncited],
                                  config=make_config(max_llm_calls=max_llm_calls))
    result = loop.run_turn("Why is the API slow?")
    total = len(client.requests) + len(pipeline.generate_pipeline.calls)
    assert total == result.loop["llm_calls"]
    assert total <= max_llm_calls + 1


# -- 7.4 1x equivalence -----------------------------------------------------------------

def test_one_iteration_agent_cites_the_same_tickets_as_answer():
    question = "Why are API endpoints slow with query timeouts?"
    pipeline = FakePipeline(answers=[CITED])

    base = object.__new__(SdAgenticRAGPipeline)
    base.store_enabled = True
    base.hybrid_retriever = pipeline.hybrid_retriever
    base.rerank_pipeline = None
    base.rerank_enabled = False
    base.ground_enabled = True
    base.ground_pipeline = pipeline.ground_pipeline
    base.generate_enabled = True
    base.generate_pipeline = pipeline.generate_pipeline
    base.guard_pipeline = pipeline.guard_pipeline
    base.embed_service = pipeline.embed_service
    base.observability = None
    answer_result = base.generate_answer(question, top_k=5)
    answer_ids = [c["ticket_id"] for c in answer_result["grounded_context"].citations]

    client = ScriptedChatClient([SUFFICIENT])
    config = AgentConfig().with_overrides(max_iterations=1, no_memory=True)
    result = AgenticRAGLoop(pipeline, config, llm_call=llm_from(client)).run_turn(question)

    assert result.grounded_ticket_ids == answer_ids
    assert answer_ids, "the question should ground something"
    # The plan was the heuristic "retrieve once": the Planner never called the LLM
    assert result.iterations[0].plan["planner"] == "heuristic"
    assert [c["tool"] for c in result.iterations[0].tool_calls] == ["retrieve"]
    assert client.requests[0]["messages"][0]["content"] == CRITIC_SYSTEM


# -- 7.5 refusal after a "sufficient" verdict (D16) ----------------------------------

def test_refusal_after_sufficient_verdict_returns_to_the_planner():
    loop, pipeline, client = make_loop([SUFFICIENT, SUFFICIENT], answers=[STANDARD_REFUSAL, CITED],
                                       config=make_config(planner="heuristic"))
    result = loop.run_turn("Why is the API slow?")
    assert events(result, "draft_refusal")
    assert result.loop["iterations"] == 2
    assert result.iterations[0].drafts == [{"revision": 0, "refusal": True}]
    assert result.answer == CITED and not result.refused and not result.partial
    assert events(result, "refine")


def test_refusal_on_the_last_iteration_is_returned():
    loop, pipeline, client = make_loop([SUFFICIENT], answers=[STANDARD_REFUSAL],
                                       config=make_config(planner="heuristic", max_iterations=1))
    result = loop.run_turn("Why is the API slow?")
    assert result.answer == STANDARD_REFUSAL and result.refused and not result.partial
    assert not events(result, "draft_refusal")
    assert result.loop["iterations"] == 1
