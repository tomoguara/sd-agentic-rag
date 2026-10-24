"""Golden-set evaluation: data integrity, metrics, evaluator and the evaluate command (tasks 10.1-10.5)."""

import json
import sys

import pytest

from agent_fakes import FakePipeline, ScriptedChatClient, llm_from

import sd_agentic_rag.agent.loop as loop_module
from sd_agentic_rag import cli
from sd_agentic_rag.agent.tools import TICKET_ID_RE
from sd_agentic_rag.rag_commands.rag_evaluate import RagEvaluateCommand
from sd_agentic_rag.sd_agentic_eval import (
    GOLDEN_TAGS,
    GoldenRunOutput,
    GoldenSetEvaluator,
    citation_precision,
    citation_recall,
    load_golden_set,
    refusal_correct,
    validate_golden_set,
)
from sd_agentic_rag.sd_agentic_ingest import SdAgenticIngest
from sd_agentic_rag.sd_agentic_guard import STANDARD_REFUSAL


GOLDEN = "data/eval/agentic_golden.json"


def merged_ticket_ids():
    with open("data/merged_tickets.json") as f:
        return {t["ticket_id"] for t in json.load(f)["tickets"]}


# -- 10.1 / 10.2 data -------------------------------------------------------------

def test_demo_tickets_load_and_are_merged():
    demo = SdAgenticIngest().load_ticket("data/agentic_demo_tickets.json")
    assert 12 <= len(demo) <= 15
    assert all(TICKET_ID_RE.fullmatch(t["ticket_id"]) for t in demo)
    assert {t["ticket_id"] for t in demo} <= merged_ticket_ids()
    statuses = {t["status"] for t in demo}
    assert len(statuses) >= 4 and len({t["sla_minutes"] for t in demo}) >= 5
    follow_ups = [t for t in demo if "follow-up to" in t["title"].lower()]
    assert len(follow_ups) >= 2
    for follow_up in follow_ups:
        [parent] = TICKET_ID_RE.findall(follow_up["title"])
        assert parent in {t["ticket_id"] for t in demo}


def test_golden_set_integrity():
    entries = load_golden_set(GOLDEN)
    assert 18 <= len(entries) <= 25
    assert validate_golden_set(entries, merged_ticket_ids()) == []
    assert {e["tag"] for e in entries} == set(GOLDEN_TAGS)
    for e in entries:
        assert (e["tag"] == "unanswerable") == (e["expected_ticket_ids"] == [])


def test_golden_validation_catches_problems():
    errors = validate_golden_set([
        {"id": "a", "query": "q", "tag": "3x", "expected_ticket_ids": []},
        {"id": "a", "query": "q", "tag": "1x", "expected_ticket_ids": ["NOPE000000"]},
        {"id": "b", "query": "q", "tag": "unanswerable", "expected_ticket_ids": ["LAT001234"]},
    ], ticket_ids={"LAT001234"})
    assert any("tag must be" in e for e in errors)
    assert any("duplicate" in e for e in errors)
    assert any("NOPE000000" in e for e in errors)
    assert any("unanswerable" in e for e in errors)


# -- 10.4 metrics and evaluator -------------------------------------------------------

def test_metric_definitions():
    assert citation_recall(["A"], ["A", "B"]) == 0.5
    assert citation_precision(["A"], ["A", "B"]) == 1.0
    assert citation_precision(["A", "C"], ["A"]) == 0.5
    assert citation_precision([], []) == 1.0           # nothing cited, nothing expected
    assert citation_precision([], ["A"]) is None
    assert citation_precision(["A"], []) == 0.0
    assert citation_recall([], []) is None
    assert refusal_correct(True, "unanswerable") and refusal_correct(False, "2x")
    assert not refusal_correct(False, "unanswerable") and not refusal_correct(True, "1x")


def output(cited, refused=False, **kwargs):
    return GoldenRunOutput(answer="a", cited_ticket_ids=cited, refused=refused, **kwargs)


ENTRIES = [
    {"id": "q1", "query": "two parts", "tag": "2x", "expected_ticket_ids": ["A", "B"]},
    {"id": "q2", "query": "one part", "tag": "1x", "expected_ticket_ids": ["C"]},
    {"id": "q3", "query": "nothing", "tag": "unanswerable", "expected_ticket_ids": []},
]


def test_evaluator_per_query_aggregates_and_deltas():
    baseline = {"two parts": output(["A"]), "one part": output(["C"]), "nothing": output([], refused=False)}
    agent = {"two parts": output(["A", "B"], iterations=2, llm_calls=5, partial=True),
             "one part": output(["C", "D"]), "nothing": output([], refused=True)}
    result = GoldenSetEvaluator({"baseline": baseline.get, "agent": agent.get}).evaluate(ENTRIES)

    q1 = result["per_query"][0]["results"]
    assert q1["baseline"]["recall"] == 0.5 and q1["baseline"]["precision"] == 1.0
    assert q1["agent"]["recall"] == 1.0 and q1["agent"]["iterations"] == 2
    q3 = result["per_query"][2]["results"]
    assert q3["baseline"]["refusal_correct"] is False and q3["agent"]["refusal_correct"] is True
    assert q3["agent"]["precision"] == 1.0

    agg = result["aggregates"]
    assert set(agg["agent"]) == {"1x", "2x", "unanswerable", "all"}
    assert agg["agent"]["2x"]["partial_rate"] == 1.0 and agg["agent"]["2x"]["llm_calls"] == 5
    assert agg["agent"]["1x"]["precision"] == 0.5
    assert result["deltas"]["2x"]["recall"] == 0.5
    assert result["deltas"]["1x"]["precision"] == -0.5
    assert result["deltas"]["unanswerable"]["refusal_correct"] == 1.0


def test_runner_error_is_recorded_and_excluded():
    def boom(query):
        raise RuntimeError("store down")
    result = GoldenSetEvaluator({"agent": boom}).evaluate(ENTRIES[:1])
    assert result["per_query"][0]["results"]["agent"] == {"error": "store down"}
    assert result["aggregates"]["agent"]["all"]["errors"] == 1
    assert "deltas" not in result


# -- 10.5 evaluate command --------------------------------------------------------------

def run_cli(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["sd-agentic-rag", "evaluate", *argv])
    with pytest.raises(SystemExit) as e:
        cli.main()
    out = capsys.readouterr()
    return e.value.code, out.out, out.err


@pytest.mark.parametrize("argv, message", [
    ([], "one of the arguments"),
    (["--prompt", "x", "--golden-file", GOLDEN], "not allowed"),
    (["--prompt", "x", "--compare"], "require --golden-file"),
    (["--prompt", "x", "--export", "r.json"], "require --golden-file"),
    (["--golden-file", GOLDEN, "--agentic"], "--prompt only"),
])
def test_argument_validation(monkeypatch, capsys, argv, message):
    code, _, err = run_cli(monkeypatch, capsys, *argv)
    assert code == 2 and message in err


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """Heuristic agent over the fake pipeline; golden set of three queries."""
    config = tmp_path / "config.yaml"
    config.write_text("agent:\n  planner: {mode: heuristic}\n  critic: {mode: heuristic}\n")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"queries": [
        {"id": "q1", "query": "Why is the API slow?", "tag": "1x", "expected_ticket_ids": ["LAT001234"]},
        {"id": "q2", "query": "Is LAT001234 resolved and how was replication lag fixed?", "tag": "2x",
         "expected_ticket_ids": ["LAT001234", "LAT002345"]},
        {"id": "q3", "query": "How do I configure the coffee machine?", "tag": "unanswerable",
         "expected_ticket_ids": []},
    ]}))
    monkeypatch.setattr(loop_module, "json_call", llm_from(ScriptedChatClient([])))
    pipeline = FakePipeline(answers=["The latency was fixed by adding a composite index. [CITATION:LAT001234]"])
    return {"config": str(config), "golden": str(golden), "pipeline": pipeline, "tmp": tmp_path}


def test_compare_table_and_export(offline, monkeypatch):
    export = offline["tmp"] / "out" / "results.json"
    monkeypatch.setenv("USE_OLLAMA_MODELS", "true")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "chat-x")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "embed-y")
    import sd_agentic_rag.model_selector as ms
    monkeypatch.setattr(ms, "_model_selector", None)

    command = RagEvaluateCommand()
    result = command.run(golden_file=offline["golden"], compare=True, export=str(export),
                         config_path=offline["config"], pipeline=offline["pipeline"])
    assert "error" not in result, result
    assert set(result["aggregates"]) == {"baseline", "agent"}
    assert set(result["deltas"]) == {"1x", "2x", "unanswerable", "all"}

    table = command.format_output(result)
    assert "| q1 | 1x | baseline |" in table and "| q1 | 1x | agent |" in table
    assert "Recall (base / agent / Δ)" in table
    assert "| unanswerable | 1 |" in table

    exported = json.loads(export.read_text())
    assert len(exported["per_query"]) == 3 and "aggregates" in exported
    assert exported["provider"] == "ollama"
    assert exported["chat_model"] == "chat-x" and exported["embedding_model"] == "embed-y"
    agent_q3 = exported["per_query"][2]["results"]["agent"]
    assert agent_q3["refused"] and agent_q3["refusal_correct"]


def test_golden_run_defaults_to_the_agent_only(offline):
    result = RagEvaluateCommand().run(golden_file=offline["golden"], config_path=offline["config"],
                                      pipeline=offline["pipeline"])
    assert list(result["aggregates"]) == ["agent"] and "deltas" not in result
    table = RagEvaluateCommand().format_output(result)
    assert "| agent | 1x |" in table and "| agent | all |" in table


def test_single_query_through_the_agent(offline):
    result = RagEvaluateCommand().run(prompt="Why is the API slow?", agentic=True,
                                      config_path=offline["config"], pipeline=offline["pipeline"])
    assert "error" not in result, result
    for key in ("predicted_answer", "relevance_score", "context_precision", "latency_ms", "model", "timestamp"):
        assert key in result
    assert result["loop"]["iterations"] >= 1 and result["agentic"] is True
    assert "## Agent Loop" in RagEvaluateCommand().format_output(result)
