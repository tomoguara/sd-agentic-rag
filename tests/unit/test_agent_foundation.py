"""Agent foundation: package, AgentConfig, state, BudgetTracker, json_call, TraceRecorder."""

import json
import subprocess
import sys

import pytest

from agent_fakes import ScriptedChatClient

from sd_agentic_rag.agent.budget import BudgetExceeded, BudgetTracker
from sd_agentic_rag.agent.config import AgentConfig, AgentConfigError, BudgetConfig
from sd_agentic_rag.agent.llm import LLMCallError, json_call, parse_json
from sd_agentic_rag.agent.state import Evidence, GateVerdict, IterationRecord, Plan, SubGoal, TurnResult
from sd_agentic_rag.agent.trace import TraceRecorder


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


# -- 1.1 package ---------------------------------------------------------

def test_agent_package_imports():
    result = subprocess.run(
        [sys.executable, "-c", "import sd_agentic_rag.agent as a; print(sorted(a.__all__))"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "AgenticRAGLoop" in result.stdout and "TurnResult" in result.stdout


# -- 1.2 AgentConfig -----------------------------------------------------

class TestAgentConfig:
    def test_defaults(self):
        config = AgentConfig.from_dict(None)
        assert config.planner.mode == "llm" and config.planner.max_sub_goals == 4
        assert config.critic.mode == "llm" and config.critic.min_similarity == 0.5
        assert config.self_review.mode == "heuristic"
        assert config.self_review.min_cited_sentence_ratio == 0.5
        assert (config.budget.max_iterations, config.budget.max_llm_calls, config.budget.max_total_tokens,
                config.budget.max_wall_seconds, config.budget.max_revisions) == (3, 12, 20000, 120, 1)
        assert config.memory.db_path == "data/sd_agentic_memory.sqlite"
        assert config.memory.long_term.top_k == 3 and config.memory.long_term.min_similarity == 0.75

    def test_project_config_yaml_section_matches_defaults(self):
        assert AgentConfig.load("config/config.yaml").to_dict() == AgentConfig().to_dict()

    def test_yaml_values_and_missing_file(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("agent:\n  critic:\n    mode: both\n  memory:\n    long_term:\n      top_k: 7\n")
        config = AgentConfig.load(str(path))
        assert config.critic.mode == "both"
        assert config.memory.long_term.top_k == 7
        assert config.memory.recent_turns == 3
        assert AgentConfig.load(str(tmp_path / "missing.yaml")).to_dict() == AgentConfig().to_dict()

    def test_cli_overrides(self):
        config = AgentConfig().with_overrides(max_iterations=1, top_k=9, no_memory=True, db_path="/tmp/x.sqlite")
        assert config.budget.max_iterations == 1
        assert config.top_k == 9
        assert config.memory.enabled is False
        assert config.memory.db_path == "/tmp/x.sqlite"
        # None leaves values unchanged
        assert AgentConfig().with_overrides().to_dict() == AgentConfig().to_dict()

    @pytest.mark.parametrize("data, key", [
        ({"critic": {"mode": "vibes"}}, "critic.mode"),
        ({"planner": {"mode": "random"}}, "planner.mode"),
        ({"budget": {"max_iterations": 0}}, "budget.max_iterations"),
        ({"budget": {"max_wall_seconds": 0}}, "budget.max_wall_seconds"),
        ({"self_review": {"min_cited_sentence_ratio": 1.5}}, "self_review.min_cited_sentence_ratio"),
        ({"budget": {"max_llm_calls": "many"}}, "budget.max_llm_calls"),
        ({"budget": {"unknown": 1}}, "agent.budget"),
    ])
    def test_invalid_values(self, data, key):
        with pytest.raises(AgentConfigError, match=key.replace(".", r"\.")):
            AgentConfig.from_dict(data)

    def test_invalid_override(self):
        with pytest.raises(AgentConfigError, match="max_iterations"):
            AgentConfig().with_overrides(max_iterations=0)


# -- 1.3 state -----------------------------------------------------------

def test_state_round_trips_through_json():
    goal = SubGoal(id="g1", query="status?", tool="lookup", ticket_id="LAT001234")
    plan = Plan(sub_goals=[goal], planner="heuristic")
    evidence = Evidence(point_id="p1", ticket_id="LAT001234", text="Status: Resolved\ntext",
                        metadata={"status": "Resolved", "tags": ["a"]}, tool="lookup",
                        scores={"score": 1.0}, vector=[0.1, 0.2])
    verdict = GateVerdict(gate="critic", passed=False, missing=["g1"], refine_hint="look up SLA")
    record = IterationRecord(iteration=1, plan=plan.to_dict(), critic=verdict.to_dict())
    result = TurnResult(question="q", answer="a", citations=[], grounded=[], partial=False, refused=True,
                        stop_reason="answered", guard={"allowed": True}, loop={"iterations": 1},
                        iterations=[record], trace=[{"event": "plan"}])
    for obj in (goal, plan, evidence, verdict, record, result):
        data = obj.to_dict()
        assert json.loads(json.dumps(data)) == data
    assert "vector" not in evidence.to_dict()
    assert plan.to_dict()["open_ids"] == ["g1"]
    assert result.to_dict()["iterations"][0]["critic"]["missing"] == ["g1"]


def test_evidence_to_ground_result_shape():
    evidence = Evidence(point_id="p1", ticket_id="T", text="body", metadata={"status": "Open"},
                        tool="retrieve", retriever="bm25", scores={"fused_score": 0.03})
    result = evidence.to_ground_result()
    assert result["payload"] == {"status": "Open", "ticket_id": "T", "chunk_text": "body"}
    assert result["fused_score"] == 0.03 and result["retriever"] == "bm25"


# -- 1.4 BudgetTracker -----------------------------------------------------

class TestBudget:
    def test_iterations(self):
        budget = BudgetTracker(BudgetConfig(max_iterations=2))
        assert budget.start_iteration() == 1
        assert budget.start_iteration() == 2
        with pytest.raises(BudgetExceeded) as e:
            budget.start_iteration()
        assert e.value.reason == "max_iterations"

    def test_llm_calls(self):
        budget = BudgetTracker(BudgetConfig(max_llm_calls=2))
        for _ in range(2):
            budget.check_llm_call()
            budget.record_llm_call(10)
        with pytest.raises(BudgetExceeded, match="max_llm_calls"):
            budget.check_llm_call()

    def test_tokens(self):
        budget = BudgetTracker(BudgetConfig(max_total_tokens=100))
        budget.record_llm_call(100)
        with pytest.raises(BudgetExceeded, match="max_total_tokens"):
            budget.check()

    def test_wall_clock(self):
        clock = FakeClock()
        budget = BudgetTracker(BudgetConfig(max_wall_seconds=5), clock=clock)
        budget.check()
        clock.now = 5.0
        with pytest.raises(BudgetExceeded, match="max_wall_seconds"):
            budget.check_llm_call()

    def test_revisions(self):
        budget = BudgetTracker(BudgetConfig(max_revisions=1))
        budget.start_iteration()
        assert budget.can_revise()
        budget.record_revision()
        assert not budget.can_revise()
        with pytest.raises(BudgetExceeded, match="max_revisions"):
            budget.record_revision()
        budget.start_iteration()            # revisions are per iteration
        assert budget.can_revise()

    def test_reserved_call_usable_exactly_once_after_exhaustion(self):
        budget = BudgetTracker(BudgetConfig(max_llm_calls=1))
        budget.record_llm_call(5)
        with pytest.raises(BudgetExceeded):
            budget.check_llm_call()
        budget.use_reserved()
        budget.record_llm_call(5)
        assert budget.llm_calls == 2 == budget.config.max_llm_calls + 1
        with pytest.raises(BudgetExceeded, match="reserved_call_used"):
            budget.use_reserved()


# -- 1.5 json_call -------------------------------------------------------

class TestJsonCall:
    def test_valid_json(self):
        client = ScriptedChatClient([{"sufficient": True}], tokens=42)
        budget = BudgetTracker()
        assert json_call("sys", "user", {"sufficient": bool}, budget, client=client, model="m") == {"sufficient": True}
        request = client.requests[0]
        assert request["temperature"] == 0
        assert request["response_format"] == {"type": "json_object"}
        assert budget.llm_calls == 1 and budget.total_tokens == 42

    def test_fenced_and_think_tagged_json(self):
        assert parse_json('<think>hmm</think>\n```json\n{"a": 1}\n```') == {"a": 1}
        assert parse_json('Here you go: {"a": 2} hope it helps') == {"a": 2}

    def test_invalid_json(self):
        budget = BudgetTracker()
        with pytest.raises(LLMCallError, match="not JSON"):
            json_call("s", "u", {"a": int}, budget, client=ScriptedChatClient(["not json"]), model="m")
        assert budget.llm_calls == 1

    @pytest.mark.parametrize("payload", [{"b": 1}, {"a": "1"}, {"a": True}])
    def test_schema_violation(self, payload):
        with pytest.raises(LLMCallError):
            json_call("s", "u", {"a": int}, BudgetTracker(), client=ScriptedChatClient([payload]), model="m")

    def test_client_error_counts_the_call(self):
        budget = BudgetTracker()
        with pytest.raises(LLMCallError, match="LLM call failed"):
            json_call("s", "u", {"a": int}, budget, client=ScriptedChatClient([RuntimeError("boom")]), model="m")
        assert budget.llm_calls == 1

    def test_budget_checked_before_call(self):
        budget = BudgetTracker(BudgetConfig(max_llm_calls=1))
        budget.record_llm_call(0)
        client = ScriptedChatClient([{"a": 1}])
        with pytest.raises(BudgetExceeded):
            json_call("s", "u", {"a": int}, budget, client=client, model="m")
        assert client.requests == []


# -- 1.6 TraceRecorder ---------------------------------------------------

class FakeMetrics:
    def __init__(self):
        self.latencies = []

    def record_latency(self, stage, ms):
        self.latencies.append((stage, ms))


class TestTrace:
    def test_events_ordered_and_sink_receives_every_event(self):
        received = []
        trace = TraceRecorder(sink=received.append)
        trace.record("planner", "plan", {"n": 1})
        trace.set_iteration(1)
        trace.record("tool", "tool_call", {"ticket_ids": ["A"]})
        trace.record("critic", "verdict", {"sufficient": True})
        assert [e["seq"] for e in trace.events] == [1, 2, 3]
        assert [e["event"] for e in trace.events] == ["plan", "tool_call", "verdict"]
        assert trace.events[0]["iteration"] == 0 and trace.events[1]["iteration"] == 1
        assert received == trace.events

    def test_previews_truncated_and_redacted(self):
        trace = TraceRecorder(preview_chars=40)
        trace.record("generate", "draft", {"preview": "Contact jane.doe@example.com " + "x" * 100,
                                           "nested": [{"note": "call 555-123-4567"}]})
        payload = trace.events[0]["payload"]
        assert "jane.doe@example.com" not in payload["preview"]
        assert "[EMAIL_REDACTED]" in payload["preview"]
        assert len(payload["preview"]) <= 43
        assert payload["nested"][0]["note"] == "call [PHONE_REDACTED]"

    def test_span_records_component_latency(self):
        metrics = FakeMetrics()
        trace = TraceRecorder(metrics=metrics)
        with trace.span("critic"):
            pass
        assert metrics.latencies[0][0] == "agent.critic"
        assert metrics.latencies[0][1] >= 0
