"""The `agent` CLI: modes, options, output and exit codes; `answer` unchanged (tasks 9.1-9.3)."""

import io
import json
import sys

import pytest

from agent_fakes import FakePipeline, ScriptedChatClient, llm_from

import sd_agentic_rag.agent.loop as loop_module
from sd_agentic_rag import cli
from sd_agentic_rag.rag_commands.rag_agent import RagAgentCommand
from sd_agentic_rag.sd_agentic_guard import STANDARD_REFUSAL


HEURISTIC_CONFIG = """
agent:
  planner: {mode: heuristic}
  critic: {mode: heuristic, min_similarity: 0.5}
"""
ORIGINAL_PIPELINE = RagAgentCommand._pipeline
CITED = "The latency was fixed by adding a composite index. [CITATION:LAT001234]"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Offline agent: fake pipeline, heuristic gates, memory in tmp_path."""
    config = tmp_path / "config.yaml"
    config.write_text(HEURISTIC_CONFIG)
    db = tmp_path / "mem.sqlite"
    monkeypatch.setenv("SD_AGENTIC_MEMORY_DB", str(db))
    answers = {"value": [CITED]}
    monkeypatch.setattr(RagAgentCommand, "_pipeline", lambda self: FakePipeline(answers=list(answers["value"])))
    monkeypatch.setattr(loop_module, "json_call", llm_from(ScriptedChatClient([])))
    return {"config": str(config), "db": db, "answers": answers}


def run_cli(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["sd-agentic-rag", "agent", *argv])
    with pytest.raises(SystemExit) as e:
        cli.main()
    out = capsys.readouterr()
    return e.value.code, out.out, out.err


# -- 9.1 parser, single turn, output, exit codes ---------------------------------

@pytest.mark.parametrize("argv", [
    ["--prompt", "x", "--history"],
    ["--chat", "--audit"],
    ["--history", "--audit"],
    [],
])
def test_mode_conflicts_and_missing_mode_are_usage_errors(monkeypatch, capsys, argv):
    code, _, err = run_cli(monkeypatch, capsys, *argv)
    assert code == 2 and "usage:" in err


@pytest.mark.parametrize("argv, message", [
    (["--prompt", "x", "--max-iterations", "0"], "--max-iterations"),
    (["--prompt", "x", "--top-k", "-1"], "--top-k"),
    (["--prompt", "x", "--full"], "--full requires --audit"),
    (["--history", "--no-memory"], "--no-memory"),
    (["--chat", "--json"], "--json"),
])
def test_invalid_options_are_usage_errors(monkeypatch, capsys, argv, message):
    code, _, err = run_cli(monkeypatch, capsys, *argv)
    assert code == 2 and message in err


def test_json_output_keys_and_exit_zero(env, monkeypatch, capsys):
    code, out, _ = run_cli(monkeypatch, capsys, "--prompt", "Why is the API slow?", "--json",
                           "--config", env["config"], "--no-memory")
    assert code == 0
    data = json.loads(out)
    assert {"answer", "citations", "partial", "guard", "loop", "trace"} <= set(data)
    assert data["answer"] == CITED and data["citations"][0]["ticket_id"] == "LAT001234"
    assert data["loop"]["stop_reason"] == "answered"
    assert not env["db"].exists()


def test_markdown_and_verbose_trace(env, monkeypatch, capsys):
    code, out, _ = run_cli(monkeypatch, capsys, "--prompt", "Why is the API slow?", "--verbose",
                           "--config", env["config"], "--no-memory")
    assert code == 0
    assert "## Answer" in out and "## Citations" in out and "tool: retrieve" in out
    assert "## Guard" in out and "Stop reason: answered" in out
    assert "## Trace" in out and "### Iteration 1" in out and "Critic (heuristic): sufficient" in out


def test_verbose_shows_both_iterations_in_order(env, monkeypatch, capsys):
    env["answers"]["value"] = ["The latency was fixed by adding a composite index on the main path today.", CITED]
    config = env["config"].replace("config.yaml", "strict.yaml")
    with open(config, "w") as f:
        f.write("agent:\n  planner: {mode: heuristic}\n  critic: {mode: heuristic}\n  budget: {max_revisions: 0}\n")
    code, out, _ = run_cli(monkeypatch, capsys, "--prompt", "Why is the API slow?", "-v", "--config", config,
                           "--no-memory")
    assert code == 0
    assert out.index("### Iteration 1") < out.index("Self-Review (heuristic): failed") < out.index("### Iteration 2")
    assert "Refined plan" in out


def test_refusal_exits_zero(env, monkeypatch, capsys):
    code, out, _ = run_cli(monkeypatch, capsys, "--prompt", "How do I configure the coffee machine?",
                           "--config", env["config"], "--no-memory")
    assert code == 0 and STANDARD_REFUSAL in out


def test_store_error_exits_non_zero(env, monkeypatch, capsys):
    import sd_agentic_rag

    class NoStorePipeline:
        """SdAgenticRAGPipeline when Qdrant is unreachable: it warns and disables the store."""
        def __init__(self):
            print("Warning: Could not initialize vector store: connection refused")
            self.store_enabled = False
            self.hybrid_retriever = None

    monkeypatch.setattr(RagAgentCommand, "_pipeline", ORIGINAL_PIPELINE)
    monkeypatch.setattr(sd_agentic_rag, "SdAgenticRAGPipeline", NoStorePipeline)
    code, out, err = run_cli(monkeypatch, capsys, "--prompt", "Why is the API slow?", "--json",
                             "--config", env["config"])
    assert code != 0
    assert "Vector store is not available" in json.loads(out)["error"]     # stdout stays valid JSON
    assert "Could not initialize vector store" in err


# -- 9.2 chat, history, audit -----------------------------------------------------

def test_chat_reads_stdin_ignores_blank_lines_and_stops_at_exit(env):
    out = io.StringIO()
    command = RagAgentCommand(input_stream=io.StringIO("Why is the API slow?\n\n   \nWhat was its resolution?\nexit\n"
                                                       "never asked\n"),
                              output_stream=out)
    result = command.run(mode="chat", thread_id="c1", config_path=env["config"])
    assert result == {"mode": "chat", "thread_id": "c1", "turns": 2}
    assert out.getvalue().count("# Agent Answer") == 2
    history = command.run(mode="history", thread_id="c1", config_path=env["config"])
    assert [t["question"] for t in history["turns"]] == ["Why is the API slow?", "What was its resolution?"]


def test_chat_ends_at_end_of_input_and_new_applies_to_first_turn(env):
    command = RagAgentCommand(input_stream=io.StringIO("Why is the API slow?\n"), output_stream=io.StringIO())
    command.run(mode="turn", prompt="Why is the API slow?", thread_id="c2", config_path=env["config"])
    command.input_stream = io.StringIO("Why is the API slow?\nquit\n")
    command.run(mode="chat", thread_id="c2", new=True, config_path=env["config"])
    assert len(command.run(mode="history", thread_id="c2", config_path=env["config"])["turns"]) == 1


def test_history_of_unknown_thread(env, monkeypatch, capsys):
    code, out, _ = run_cli(monkeypatch, capsys, "--history", "--thread-id", "nobody", "--config", env["config"])
    assert code == 0
    assert "No conversation is saved for thread `nobody`" in out
    assert not env["db"].exists()             # reading history does not create the database


def test_audit_across_two_conversations(env, monkeypatch, capsys):
    for argv in (["--prompt", "Why is the API slow?"], ["--new", "--prompt", "VPN keeps dropping?"]):
        code, _, _ = run_cli(monkeypatch, capsys, *argv, "--thread-id", "t1", "--config", env["config"])
        assert code == 0
    code, out, _ = run_cli(monkeypatch, capsys, "--audit", "--thread-id", "t1", "--json", "--config", env["config"])
    steps = json.loads(out)["steps"]
    assert {s["turn_no"] for s in steps} == {1, 2}
    assert {s["epoch"] for s in steps} == {0, 1}
    assert [s["turn_no"] for s in steps] == sorted(s["turn_no"] for s in steps)

    code, out, _ = run_cli(monkeypatch, capsys, "--audit", "--thread-id", "t1", "--config", env["config"])
    assert "## Turn 1 (conversation 0)" in out and "## Turn 2 (conversation 1)" in out
    truncated = max(len(line) for line in out.splitlines())
    code, full_out, _ = run_cli(monkeypatch, capsys, "--audit", "--full", "--thread-id", "t1", "--config", env["config"])
    assert max(len(line) for line in full_out.splitlines()) > truncated

    code, out, _ = run_cli(monkeypatch, capsys, "--history", "--thread-id", "t1", "--config", env["config"])
    assert "VPN keeps dropping?" in out and "Why is the API slow?" not in out


# -- 9.3 answer unchanged -----------------------------------------------------------

def test_answer_does_not_create_the_memory_database(tmp_path, monkeypatch):
    import sd_agentic_rag
    from types import SimpleNamespace
    from sd_agentic_rag.rag_commands.rag_answer import RagAnswerCommand

    db = tmp_path / "mem.sqlite"
    monkeypatch.setenv("SD_AGENTIC_MEMORY_DB", str(db))
    monkeypatch.chdir(tmp_path)

    class StubPipeline:
        def generate_answer(self, query_text, top_k, ground_enabled):
            grounded = SimpleNamespace(citations=[{"ticket_id": "LAT001234"}])
            return {"answer": SimpleNamespace(answer=CITED, citations=[]), "grounded_context": grounded,
                    "guard_result": {"allowed": True}}

        def get_observability_metrics(self):
            return {}

    monkeypatch.setattr(sd_agentic_rag, "SdAgenticRAGPipeline", StubPipeline)
    result = RagAnswerCommand().run(prompt="Why is the API slow?")
    assert result["answer"] == CITED
    assert not db.exists() and not (tmp_path / "data").exists()
