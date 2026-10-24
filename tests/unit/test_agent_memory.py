"""Agent memory: schema, threads, audit, past resolutions, loop wiring (tasks 8.1-8.5)."""

import sqlite3

import pytest

from agent_fakes import FakePipeline, ScriptedChatClient, llm_from

from sd_agentic_rag.agent.config import AgentConfig
from sd_agentic_rag.agent.loop import AgenticRAGLoop
from sd_agentic_rag.agent.memory import DB_ENV_VAR, SCHEMA_VERSION, AgentMemory, resolve_db_path


# -- 8.1 schema and path ---------------------------------------------------------

def test_schema_created_on_first_use(tmp_path):
    path = tmp_path / "nested" / "mem.sqlite"
    assert not path.exists()
    memory = AgentMemory(str(path))
    assert path.exists()
    tables = {r[0] for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"threads", "turns", "steps", "resolutions"} <= tables
    assert memory.schema_version == SCHEMA_VERSION
    memory.close()
    AgentMemory(str(path)).close()          # re-opening an existing file is fine


def test_path_resolution_order(monkeypatch, tmp_path):
    monkeypatch.delenv(DB_ENV_VAR, raising=False)
    assert resolve_db_path(None, None) == "data/sd_agentic_memory.sqlite"
    assert resolve_db_path(None, "cfg.sqlite") == "cfg.sqlite"
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "env.sqlite"))
    assert resolve_db_path(None, "cfg.sqlite") == str(tmp_path / "env.sqlite")
    assert resolve_db_path(str(tmp_path / "cli.sqlite"), "cfg.sqlite") == str(tmp_path / "cli.sqlite")


def test_memory_database_is_git_ignored():
    with open(".gitignore") as f:
        assert "data/*.sqlite" in f.read().split()


# -- 8.2 thread turns -------------------------------------------------------------

def test_follow_up_loads_across_separate_instances(tmp_path):
    path = str(tmp_path / "m.sqlite")
    with AgentMemory(path) as first:
        first.save_turn("t1", first.next_turn_no("t1"), "Why is the API slow?", "Missing index [CITATION:LAT001234]",
                        ["LAT001234"])
    with AgentMemory(path) as second:          # a separate process would open the file the same way
        turns = second.recent_turns("t1", 3)
        assert [t["cited_ticket_ids"] for t in turns] == [["LAT001234"]]
        assert turns[0]["question"] == "Why is the API slow?"
        assert second.next_turn_no("t1") == 2


def test_threads_are_isolated(tmp_path):
    with AgentMemory(str(tmp_path / "m.sqlite")) as memory:
        memory.save_turn("t1", 1, "q", "a", ["LAT001234"])
        assert memory.recent_turns("t2", 3) == []
        assert memory.history("t2") == []


def test_recent_turns_limit_and_order(tmp_path):
    with AgentMemory(str(tmp_path / "m.sqlite")) as memory:
        for n in range(1, 6):
            memory.save_turn("t", n, f"q{n}", f"a{n}", [])
        assert [t["question"] for t in memory.recent_turns("t", 2)] == ["q4", "q5"]


def test_new_conversation_hides_but_keeps_earlier_turns(tmp_path):
    with AgentMemory(str(tmp_path / "m.sqlite")) as memory:
        memory.save_turn("t1", 1, "old question", "old answer", ["LAT001234"])
        memory.append_step("t1", 1, {"seq": 1, "iteration": 1, "component": "planner", "event": "plan", "payload": {}})
        memory.new_conversation("t1")
        assert memory.recent_turns("t1", 3) == []
        assert memory.history("t1") == []
        memory.save_turn("t1", memory.next_turn_no("t1"), "new question", "new answer", [])
        assert [t["question"] for t in memory.history("t1")] == ["new question"]
        assert [s["turn_no"] for s in memory.audit("t1")] == [1]       # earlier steps still listed
        rows = sqlite3.connect(memory.path).execute("SELECT question FROM turns ORDER BY id").fetchall()
        assert [r[0] for r in rows] == ["old question", "new question"]


def test_pii_is_redacted_before_writing(tmp_path):
    with AgentMemory(str(tmp_path / "m.sqlite")) as memory:
        memory.save_turn("t", 1, "I am jane@example.com", "Reply to jane@example.com", [],
                         resolution={"vector": [1.0, 0.0], "embedding_model": "m"})
        memory.append_step("t", 1, {"seq": 1, "component": "guard", "event": "question",
                                    "payload": {"question": "I am jane@example.com"}})
        dump = "\n".join(sqlite3.connect(memory.path).iterdump())
        assert "jane@example.com" not in dump
        assert "[EMAIL_REDACTED]" in dump


# -- 8.3 audit -----------------------------------------------------------------

def test_two_iteration_turn_steps_stored_in_order(tmp_path):
    memory = AgentMemory(str(tmp_path / "m.sqlite"))
    client = ScriptedChatClient([
        {"sufficient": False, "covered": [], "missing": ["g1"]},
        {"sufficient": True, "covered": ["g1"], "missing": []},
    ])
    config = AgentConfig()
    config.planner.mode = "heuristic"
    loop = AgenticRAGLoop(FakePipeline(), config, memory=memory, thread_id="a", llm_call=llm_from(client))
    result = loop.run_turn("Why is the API slow?")
    assert result.loop["iterations"] == 2
    audit = memory.audit("a")
    assert [s["seq"] for s in audit] == list(range(1, len(audit) + 1))
    assert len(audit) == len(result.trace)
    sequence = [(s["iteration"], s["event"]) for s in audit if s["event"] in
                ("plan", "tool_call", "verdict", "draft", "refine")]
    assert sequence[:4] == [(0, "plan"), (1, "tool_call"), (1, "verdict"), (1, "refine")]
    assert (2, "tool_call") in sequence and (2, "draft") in sequence
    assert sequence.index((2, "tool_call")) < sequence.index((2, "draft"))
    tool_steps = [s for s in audit if s["event"] == "tool_call"]
    assert tool_steps[0]["payload"]["evidence_ticket_ids"]


def test_no_api_updates_or_deletes_steps():
    names = [n for n in dir(AgentMemory) if not n.startswith("__")]
    assert not [n for n in names if any(word in n.lower() for word in ("update", "delete", "remove", "clear", "edit"))]


# -- 8.4 past resolutions --------------------------------------------------------

def test_cross_thread_hint_threshold_and_model(tmp_path):
    with AgentMemory(str(tmp_path / "m.sqlite")) as memory:
        memory.save_turn("a", 1, "DB01 replication lag?", "WAL tuning [CITATION:LAT002345]", ["LAT002345"],
                         resolution={"vector": [0.0, 1.0, 0.0], "embedding_model": "ollama-embed"})
        memory.save_turn("a", 2, "VPN drops?", "cert [CITATION:NET000111]", ["NET000111"],
                         resolution={"vector": [0.0, 0.0, 1.0], "embedding_model": "ollama-embed"})
        hits = memory.search_resolutions([0.0, 0.95, 0.1], "ollama-embed", top_k=3, min_similarity=0.75)
        assert [h["cited_ticket_ids"] for h in hits] == [["LAT002345"]]
        assert hits[0]["thread_id"] == "a"
        assert len(memory.search_resolutions([0.0, 0.7, 0.7], "ollama-embed", min_similarity=0.5)) == 2
        assert len(memory.search_resolutions([0.0, 0.7, 0.7], "ollama-embed", top_k=1, min_similarity=0.5)) == 1
        # Provider switched: entries from another embedding model are ignored
        assert memory.search_resolutions([0.0, 1.0, 0.0], "text-embedding-3-small", min_similarity=0.0) == []
        row = sqlite3.connect(memory.path).execute("SELECT embedding_model, dims, length(vector) FROM resolutions").fetchone()
        assert row == ("ollama-embed", 3, 12)          # float32 BLOB


def _memory_loop(tmp_path, answers, responses, thread="t", **config_kwargs):
    memory = AgentMemory(str(tmp_path / "m.sqlite"))
    config = AgentConfig()
    config.planner.mode = "heuristic"
    for key, value in config_kwargs.items():
        setattr(config.memory, key, value)
    pipeline = FakePipeline(answers=answers)
    return AgenticRAGLoop(pipeline, config, memory=memory, thread_id=thread,
                          llm_call=llm_from(ScriptedChatClient(responses))), memory, pipeline


def test_resolution_written_only_for_answers_that_pass(tmp_path):
    loop, memory, _ = _memory_loop(tmp_path, ["Fixed by adding a composite index. [CITATION:LAT001234]"],
                                   [{"sufficient": True, "covered": ["g1"], "missing": []}] * 3)
    loop.run_turn("Why is the API slow?")
    assert sqlite3.connect(memory.path).execute("SELECT COUNT(*) FROM resolutions").fetchone()[0] == 1


def test_refusal_is_not_stored_as_resolution(tmp_path):
    loop, memory, _ = _memory_loop(tmp_path, ["unused"], [])
    result = loop.run_turn("How do I configure the coffee machine?")
    assert result.refused
    con = sqlite3.connect(memory.path)
    assert con.execute("SELECT COUNT(*) FROM resolutions").fetchone()[0] == 0
    assert con.execute("SELECT refused FROM turns").fetchone()[0] == 1


def test_hint_from_another_thread_reaches_the_planner(tmp_path):
    loop_a, memory, _ = _memory_loop(tmp_path, ["WAL settings were tuned. [CITATION:LAT002345]"],
                                     [{"sufficient": True, "covered": ["g1"], "missing": []}] * 3, thread="a")
    loop_a.run_turn("DB01 replication lag?")
    loop_b = AgenticRAGLoop(FakePipeline(answers=["WAL settings were tuned. [CITATION:LAT002345]"]),
                            loop_a.config, memory=memory, thread_id="b",
                            llm_call=llm_from(ScriptedChatClient([{"sufficient": True, "covered": [], "missing": []}] * 3)))
    result = loop_b.run_turn("replica falling behind on DB01")
    context = [e for e in result.trace if e["event"] == "context"][0]
    assert context["payload"]["hints"] == ["LAT002345"]
    assert context["payload"]["recent_turns"] == 0           # thread b has no turns of its own
    lookups = [c for c in result.iterations[0].tool_calls if c["tool"] == "lookup"]
    assert lookups and lookups[0]["ticket_id"] == "LAT002345"


# -- 8.5 loop wiring ---------------------------------------------------------------

def test_remembered_ticket_cited_only_if_regrounded(tmp_path):
    sufficient = [{"sufficient": True, "covered": ["g1"], "missing": []}] * 5
    loop, memory, _ = _memory_loop(tmp_path, ["A composite index fixed it. [CITATION:LAT001234]"], sufficient)
    loop.run_turn("Why is the API slow?")

    # The remembered ticket has since been removed from the index
    pipeline = FakePipeline(answers=["Its resolution was an index. [CITATION:LAT001234]"],
                            points=[p for p in FakePipeline().vector_store.points if p["payload"]["ticket_id"] != "LAT001234"])
    follow_up = AgenticRAGLoop(pipeline, loop.config, memory=memory, thread_id="t",
                               llm_call=llm_from(ScriptedChatClient(sufficient)))
    result = follow_up.run_turn("What was its resolution?")
    planned = [c for r in result.iterations for c in r.tool_calls]
    assert planned[0]["tool"] == "lookup" and planned[0]["ticket_id"] == "LAT001234"
    assert planned[0]["count"] == 0
    assert "LAT001234" not in result.grounded_ticket_ids
    assert "LAT001234" not in result.answer and result.citations == []

    # When the ticket is still indexed, the follow-up re-grounds and may cite it
    again = AgenticRAGLoop(FakePipeline(answers=["Its resolution was an index. [CITATION:LAT001234]"]), loop.config,
                           memory=memory, thread_id="t", llm_call=llm_from(ScriptedChatClient(sufficient)))
    result = again.run_turn("What was its resolution?")
    assert result.cited_ticket_ids == ["LAT001234"]
    assert result.iterations[0].tool_calls[0]["tool"] == "lookup"


def test_new_conversation_flag_in_the_loop(tmp_path):
    sufficient = [{"sufficient": True, "covered": ["g1"], "missing": []}] * 5
    loop, memory, _ = _memory_loop(tmp_path, ["A composite index fixed it. [CITATION:LAT001234]"], sufficient,
                                   long_term=AgentConfig().memory.long_term.__class__(enabled=False))
    loop.run_turn("Why is the API slow?")
    result = loop.run_turn("What was its resolution?", new_conversation=True)
    assert [c["tool"] for c in result.iterations[0].tool_calls] == ["retrieve"]
    assert len(memory.history("t")) == 1 and result.turn_no == 2


def test_no_memory_never_creates_the_database(tmp_path, monkeypatch):
    path = tmp_path / "never.sqlite"
    monkeypatch.setenv(DB_ENV_VAR, str(path))
    config = AgentConfig().with_overrides(no_memory=True, db_path=str(path))
    config.planner.mode = "heuristic"
    loop = AgenticRAGLoop(FakePipeline(), config,
                          llm_call=llm_from(ScriptedChatClient([{"sufficient": True, "covered": [], "missing": []}])))
    result = loop.run_turn("Why is the API slow?")
    assert loop.memory is None and result.thread_id is None
    assert not path.exists()


# -- 8.6 follow-up relevance question (D17) -------------------------------------------

def _relevance_loop(memory, thread, answers, sufficient):
    from agent_fakes import make_guardrails, FakeEmbeddingService
    embed = FakeEmbeddingService()
    pipeline = FakePipeline(answers=answers, embedding_service=embed,
                            guardrails=make_guardrails(enable_relevance_gate=True, relevance_threshold=0.5,
                                                       embedding_service=embed))
    config = AgentConfig()
    config.planner.mode = "heuristic"
    config.memory.long_term.enabled = False
    return AgenticRAGLoop(pipeline, config, memory=memory, thread_id=thread,
                          llm_call=llm_from(ScriptedChatClient(sufficient)))


def test_follow_up_passes_relevance_with_the_previous_question(tmp_path):
    sufficient = [{"sufficient": True, "covered": ["g1"], "missing": []}] * 5
    memory = AgentMemory(str(tmp_path / "m.sqlite"))
    first = _relevance_loop(memory, "t", ["Slow queries were fixed by an index. [CITATION:LAT001234]"], sufficient)
    assert first.run_turn("Why is the API slow?").guard["allowed"]

    follow_up = _relevance_loop(memory, "t", ["Its resolution was a composite index. [CITATION:LAT001234]"], sufficient)
    result = follow_up.run_turn("What was its resolution?")
    assert result.iterations[0].tool_calls[0]["ticket_id"] == "LAT001234"
    assert result.guard["allowed"] and not result.refused
    assert result.cited_ticket_ids == ["LAT001234"]
    assert memory.history("t")[-1]["question"] == "What was its resolution?"   # stored turn keeps the real question

    # Control: the bare follow-up alone fails the relevance gate (similarity 0 with the fake embeddings)
    bare = follow_up.pipeline.guard_pipeline.check_relevance(
        "What was its resolution?", result.answer, follow_up.pipeline.embed_service)
    assert not bare["relevant"]


def test_relevance_question_only_for_remembered_ids_not_named():
    from sd_agentic_rag.agent.planner import PlannerContext
    from sd_agentic_rag.agent.state import IterationRecord
    context = PlannerContext(recent_turns=[{"question": "Why is the API slow?", "answer": "a",
                                            "cited_ticket_ids": ["LAT001234"]}])
    looked_up = [IterationRecord(iteration=1, plan={}, tool_calls=[{"tool": "lookup", "ticket_id": "LAT001234"}])]
    assert AgenticRAGLoop._relevance_question("What was its resolution?", context, looked_up) == \
        "Why is the API slow? What was its resolution?"
    assert AgenticRAGLoop._relevance_question("What fixed LAT001234?", context, looked_up) is None
    retrieved = [IterationRecord(iteration=1, plan={}, tool_calls=[{"tool": "retrieve", "ticket_id": None}])]
    assert AgenticRAGLoop._relevance_question("How do I reset a password?", context, retrieved) is None
    assert AgenticRAGLoop._relevance_question("What was its resolution?", PlannerContext(), looked_up) is None
