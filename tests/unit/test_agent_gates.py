"""Guard wrapper, Planner, Critic and Self-Review (tasks 3.1, 4.1-4.4, 5.1-5.2, 6.1-6.2)."""

from types import SimpleNamespace

import pytest

from agent_fakes import ScriptedChatClient, llm_from, make_guardrails

from sd_agentic_rag.agent.budget import BudgetTracker
from sd_agentic_rag.agent.config import CriticConfig, PlannerConfig, SelfReviewConfig
from sd_agentic_rag.agent.critic import Critic
from sd_agentic_rag.agent.grounding import EvidencePool, ground_pool
from sd_agentic_rag.agent.guard import AgentGuard
from sd_agentic_rag.agent.planner import HeuristicPlanner, Planner, PlannerContext
from sd_agentic_rag.agent.self_review import SelfReview, cited_sentence_ratio
from sd_agentic_rag.agent.state import Evidence, GateVerdict, Plan, SubGoal
from sd_agentic_rag.agent.trace import TraceRecorder
from sd_agentic_rag.sd_agentic_ground import ContextGrounding, GroundedContext
from sd_agentic_rag.sd_agentic_guard import STANDARD_REFUSAL
from sd_agentic_rag.sd_agentic_retrieve import HybridRetriever


KNOWN = {
    "service": ["Database Services", "Network"],
    "severity": ["Critical", "High", "Low", "Medium"],
    "ticket_type": ["Incident", "Performance Issue"],
    "status": ["Closed", "Open", "Resolved"],
}


def grounded_for(*ticket_ids) -> GroundedContext:
    citations = [{"citation_id": f"[CITATION:{t}]", "ticket_id": t, "score": 1.0, "rank": i + 1,
                  "retriever": "vector", "content_preview": ""} for i, t in enumerate(ticket_ids)]
    text = "\n---\n".join(f"[CITATION:{t}]\nStatus: Resolved\npassage about {t}\n" for t in ticket_ids)
    return GroundedContext(context_text=text, citations=citations, relevance_scores=[1.0] * len(ticket_ids),
                           total_tokens=len(text) // 4)


# -- 3.1 Guard -----------------------------------------------------------

class TestAgentGuard:
    def test_email_in_evidence_never_reaches_an_llm_prompt(self):
        guard = AgentGuard(make_guardrails())
        raw = Evidence(point_id="p1", ticket_id="LAT001234", text="Reported by jane.doe@example.com: slow API",
                       metadata={"assigned_to": "ops@example.com", "tags": ["x"]}, tool="retrieve")
        [clean] = guard.redact_evidence([raw])
        assert "jane.doe@example.com" not in clean.text and "[EMAIL_REDACTED]" in clean.text
        assert clean.metadata["assigned_to"] == "[EMAIL_REDACTED]"
        assert "jane.doe" in raw.text          # the original is not mutated

        pool = EvidencePool()
        pool.add("g1", [clean])
        grounded = ground_pool(ContextGrounding(max_passages=5, max_tokens=4000), pool, {"g1": "q"})
        client = ScriptedChatClient([{"sufficient": True, "covered": ["g1"], "missing": []}])
        plan = Plan(sub_goals=[SubGoal(id="g1", query="slow API", tool="retrieve")])
        Critic(CriticConfig(mode="llm"), llm_call=llm_from(client)).judge("slow API?", plan, grounded, pool,
                                                                          budget=BudgetTracker())
        assert client.requests, "the critic should have consulted the LLM"
        assert not any("jane.doe@example.com" in p for p in client.prompts)
        assert any("[EMAIL_REDACTED]" in p for p in client.prompts)

    def test_finalize_refuses_a_fabricated_citation_when_enforced(self):
        guard = AgentGuard(make_guardrails(enforce=True))
        result = guard.finalize("Fixed by a reboot. [CITATION:FAKE000001]", "how fixed?", grounded_for("LAT001234"))
        assert result["allowed"] is False and result["replaced"] is True
        assert result["modified_answer"] == STANDARD_REFUSAL

    def test_finalize_report_only_when_not_enforced(self):
        guard = AgentGuard(make_guardrails(enforce=False))
        result = guard.finalize("Fixed by a reboot. [CITATION:FAKE000001]", "how fixed?", grounded_for("LAT001234"))
        assert result["allowed"] is False and result["replaced"] is False

    def test_correct_refusal_passes(self):
        guard = AgentGuard(make_guardrails(enforce=True))
        result = guard.finalize(STANDARD_REFUSAL, "unrelated?", grounded_for())
        assert result["allowed"] is True and result["replaced"] is False
        assert guard.is_refusal(result["modified_answer"])

    def test_question_redaction(self):
        assert AgentGuard(make_guardrails()).redact("mail me at a.b@c.com") == "mail me at [EMAIL_REDACTED]"


# -- 4.x Planner -------------------------------------------------------------

class TestHeuristicPlanner:
    planner = HeuristicPlanner(PlannerConfig(), top_k=5)

    def test_plain_question_is_exactly_one_retrieve(self):
        plan = self.planner.plan("Why are API calls slow?", PlannerContext(known_values=KNOWN))
        assert [(g.tool, g.query) for g in plan.sub_goals] == [("retrieve", "Why are API calls slow?")]
        assert plan.planner == "heuristic"

    def test_ticket_ids_become_lookups(self):
        plan = self.planner.plan("Is LAT001234 resolved and how does it compare to NET000111?",
                                 PlannerContext(known_values=KNOWN))
        assert [(g.tool, g.ticket_id) for g in plan.sub_goals] == [("lookup", "LAT001234"), ("lookup", "NET000111")]

    def test_known_value_becomes_filtered_retrieve(self):
        plan = self.planner.plan("Recent incident tickets in database services?", PlannerContext(known_values=KNOWN))
        [g] = plan.sub_goals
        assert g.tool == "filtered_retrieve"
        assert g.filters == {"service": "Database Services", "ticket_type": "Incident"}

    def test_status_values_alone_do_not_filter(self):
        plan = self.planner.plan("Which login problems were resolved?", PlannerContext(known_values=KNOWN))
        assert [g.tool for g in plan.sub_goals] == ["retrieve"]

    def test_memory_reference_adds_lookups_only_with_a_referring_word(self):
        context = PlannerContext(known_values=KNOWN, recent_turns=[
            {"question": "slow API?", "answer": "...", "cited_ticket_ids": ["LAT001234"]}])
        follow_up = self.planner.plan("What was its resolution?", context)
        assert [(g.tool, g.ticket_id, g.source) for g in follow_up.sub_goals] == [("lookup", "LAT001234", "memory")]
        fresh = self.planner.plan("How do I reset a password?", context)
        assert [g.tool for g in fresh.sub_goals] == ["retrieve"]

    def test_hints_add_lookups_next_to_the_retrieve(self):
        plan = self.planner.plan("replica falling behind", PlannerContext(known_values=KNOWN, hints=["LAT002345"]))
        assert [(g.tool, g.ticket_id, g.source) for g in plan.sub_goals] == [
            ("retrieve", None, "question"), ("lookup", "LAT002345", "hint")]

    def test_max_sub_goals_cap(self):
        planner = HeuristicPlanner(PlannerConfig(max_sub_goals=2), top_k=5)
        plan = planner.plan("AAA000001 BBB000002 CCC000003", PlannerContext())
        assert len(plan.sub_goals) == 2

    def test_refine_drops_filters_and_doubles_top_k_capped(self):
        planner = HeuristicPlanner(PlannerConfig(max_top_k=8), top_k=5)
        plan = Plan(sub_goals=[
            SubGoal(id="g1", query="q", tool="filtered_retrieve", filters={"service": "Network"}, top_k=5),
            SubGoal(id="g2", query="q", tool="lookup", ticket_id="LAT001234", top_k=5),
        ])
        refined = planner.refine(plan, GateVerdict(gate="critic", passed=False, missing=["g1", "g2"]))
        g1 = refined.get("g1")
        assert (g1.tool, g1.filters, g1.top_k) == ("retrieve", {}, 8)
        g2 = refined.get("g2")                          # a missing lookup becomes a retrieve of ID + query
        assert (g2.tool, g2.query, g2.ticket_id, g2.top_k) == ("retrieve", "LAT001234 q", None, 5)
        assert refined.open_ids == ["g1", "g2"]
        again = planner.refine(refined, GateVerdict(gate="critic", passed=False, missing=["g1"]))
        assert again.open_sub_goals == []               # nothing left to broaden


def test_known_values_collected_once_during_the_bm25_scroll():
    points = [SimpleNamespace(id=f"p{i}", payload=payload) for i, payload in enumerate([
        {"chunk_text": "a", "service": "Network", "severity": "Low", "ticket_type": "Incident", "status": "Open"},
        {"chunk_text": "b", "service": "Database Services", "severity": "Low", "ticket_type": "Incident",
         "status": "Resolved"},
        {"chunk_text": "c", "service": "", "severity": None, "ticket_type": "Incident", "status": "Open"},
    ])]

    class Client:
        calls = 0

        def scroll(self, **kwargs):
            Client.calls += 1
            return points, None

    store = SimpleNamespace(client=Client(), collection_name="c")
    retriever = HybridRetriever(vector_store=store, embedding_service=None)
    assert Client.calls == 1
    assert retriever.known_metadata_values() == {
        "service": ["Database Services", "Network"],
        "severity": ["Low"],
        "ticket_type": ["Incident"],
        "status": ["Open", "Resolved"],
    }
    assert Client.calls == 1


class TestLLMPlanner:
    def make(self, responses, **config):
        client = ScriptedChatClient(responses)
        trace = TraceRecorder()
        planner = Planner(PlannerConfig(**config), top_k=5, llm_call=llm_from(client), trace=trace)
        return planner, client, trace

    def test_valid_two_part_plan(self):
        planner, client, trace = self.make([{"sub_goals": [
            {"id": "g1", "query": "LAT001234 status", "tool": "lookup", "ticket_id": "lat001234"},
            {"id": "g2", "query": "SLA for latency incidents", "tool": "filtered_retrieve",
             "filters": {"service": "database services"}},
        ]}])
        context = PlannerContext(known_values=KNOWN, hints=["LAT002345"],
                                 recent_turns=[{"question": "q1", "answer": "a1", "cited_ticket_ids": ["NET000111"]}])
        plan = planner.plan("Is LAT001234 resolved and what was its SLA?", context, budget=BudgetTracker())
        assert plan.planner == "llm"
        assert [(g.tool, g.ticket_id) for g in plan.sub_goals] == [("lookup", "LAT001234"), ("filtered_retrieve", None)]
        assert plan.sub_goals[1].filters == {"service": "Database Services"}   # canonical spelling
        system, user = client.prompts
        assert "lookup" in system and "filtered_retrieve" in system and "Database Services" in system
        assert "NET000111" in user and "LAT002345" in user
        assert trace.events_named("plan")

    @pytest.mark.parametrize("response, error", [
        ("this is not json", "not JSON"),
        ({"sub_goals": [{"id": "g1", "query": "x", "tool": "web_search"}]}, "web_search"),
        ({"sub_goals": [{"id": "g1", "query": "x", "tool": "filtered_retrieve",
                         "filters": {"assigned_to": "Bob"}}]}, "assigned_to"),
        ({"plan": []}, "sub_goals"),
    ])
    def test_invalid_plan_falls_back_to_heuristic(self, response, error):
        planner, _, trace = self.make([response])
        plan = planner.plan("Why are API calls slow?", PlannerContext(known_values=KNOWN), budget=BudgetTracker())
        assert plan.planner == "heuristic"
        assert [g.tool for g in plan.sub_goals] == ["retrieve"]
        [fallback] = trace.events_named("planner_fallback")
        assert error in fallback["payload"]["error"]

    def test_too_many_sub_goals_are_capped(self):
        planner, _, trace = self.make([{"sub_goals": [
            {"id": f"g{i}", "query": f"q{i}", "tool": "retrieve"} for i in range(1, 7)]}], max_sub_goals=4)
        plan = planner.plan("big question", PlannerContext(), budget=BudgetTracker())
        assert plan.planner == "llm" and [g.id for g in plan.sub_goals] == ["g1", "g2", "g3", "g4"]
        assert trace.events_named("planner_capped")

    def test_refine_with_critic_feedback(self):
        planner, client, _ = self.make([{"sub_goals": [{"id": "g2", "query": "LAT001234 SLA minutes",
                                                        "tool": "lookup", "ticket_id": "LAT001234"}]}])
        plan = Plan(planner="llm", sub_goals=[SubGoal(id="g1", query="status", tool="lookup", ticket_id="LAT001234"),
                                              SubGoal(id="g2", query="sla", tool="retrieve")])
        verdict = GateVerdict(gate="critic", passed=False, missing=["g2"], refine_hint="look for the SLA")
        refined = planner.refine("q", plan, verdict, PlannerContext(), budget=BudgetTracker())
        assert [g.id for g in refined.open_sub_goals] == ["g2"]
        assert refined.get("g2").tool == "lookup" and refined.get("g1").query == "status"
        assert "look for the SLA" in client.prompts[-1]

    def test_refine_failure_falls_back_to_heuristic_refine(self):
        planner, _, trace = self.make(["garbage"])
        plan = Plan(planner="llm", sub_goals=[SubGoal(id="g1", query="q", tool="retrieve", top_k=5)])
        refined = planner.refine("q", plan, GateVerdict(gate="critic", passed=False, missing=["g1"]),
                                 budget=BudgetTracker())
        assert refined.get("g1").top_k == 10
        assert trace.events_named("planner_fallback")

    def test_force_heuristic_never_calls_the_llm(self):
        planner, client, _ = self.make([{"sub_goals": []}])
        plan = planner.plan("Why are API calls slow?", PlannerContext(), budget=BudgetTracker(), force_heuristic=True)
        assert plan.planner == "heuristic" and client.requests == []


# -- 5.x Critic --------------------------------------------------------------

def critic_inputs():
    pool = EvidencePool()
    pool.add("g1", [Evidence(point_id="p1", ticket_id="LAT001234", text="latency", metadata={}, tool="retrieve",
                             vector=[1.0, 0.0, 0.0])])
    pool.add("g2", [Evidence(point_id="p2", ticket_id="NET000111", text="vpn", metadata={}, tool="retrieve",
                             vector=[0.0, 0.0, 1.0])])
    plan = Plan(sub_goals=[SubGoal(id="g1", query="latency", tool="retrieve"),
                           SubGoal(id="g2", query="replication", tool="retrieve")])
    cache = {"latency": [1.0, 0.0, 0.0], "replication": [0.0, 1.0, 0.0]}
    return pool, plan, cache


class TestCritic:
    def test_no_evidence_is_insufficient_without_an_llm_call(self):
        client = ScriptedChatClient([{"sufficient": True}])
        critic = Critic(CriticConfig(mode="llm"), llm_call=llm_from(client))
        plan = Plan(sub_goals=[SubGoal(id="g1", query="q", tool="retrieve")])
        verdict = critic.judge("q", plan, grounded_for(), EvidencePool(), budget=BudgetTracker())
        assert not verdict.sufficient and verdict.missing == ["g1"] and verdict.mode == "precheck"
        assert client.requests == []

    def test_heuristic_one_covered_one_uncovered(self):
        pool, plan, cache = critic_inputs()
        critic = Critic(CriticConfig(mode="heuristic", min_similarity=0.5), embedding_cache=cache)
        verdict = critic.judge("q", plan, grounded_for("LAT001234", "NET000111"), pool)
        assert not verdict.sufficient
        assert verdict.covered == ["g1"] and verdict.missing == ["g2"]

    def test_heuristic_all_covered(self):
        pool, plan, cache = critic_inputs()
        cache["replication"] = [0.0, 0.2, 1.0]
        verdict = Critic(CriticConfig(mode="heuristic"), embedding_cache=cache).judge(
            "q", plan, grounded_for("LAT001234", "NET000111"), pool)
        assert verdict.sufficient and verdict.covered == ["g1", "g2"]

    def test_heuristic_fetches_stored_vectors_and_covers_lookups_by_ticket(self):
        pool = EvidencePool()
        pool.add("g1", [Evidence(point_id="p1", ticket_id="LAT001234", text="t", metadata={}, tool="lookup")])
        plan = Plan(sub_goals=[SubGoal(id="g1", query="status", tool="lookup", ticket_id="LAT001234"),
                               SubGoal(id="g2", query="latency", tool="retrieve")])
        fetched = []

        def fetch(ids):
            fetched.extend(ids)
            return {"p1": [1.0, 0.0, 0.0]}

        verdict = Critic(CriticConfig(mode="heuristic"), fetch_vectors=fetch,
                         embedding_cache={"latency": [1.0, 0.0, 0.0]}).judge("q", plan, grounded_for("LAT001234"), pool)
        assert fetched == ["p1"] and verdict.sufficient

    def test_llm_mode(self):
        pool, plan, cache = critic_inputs()
        client = ScriptedChatClient([{"sufficient": False, "covered": ["g1"], "missing": ["g2", "zz"],
                                      "refine_hint": "search replication lag"}])
        verdict = Critic(CriticConfig(mode="llm"), llm_call=llm_from(client), embedding_cache=cache).judge(
            "q", plan, grounded_for("LAT001234"), pool, budget=BudgetTracker())
        assert verdict.mode == "llm" and not verdict.sufficient
        assert verdict.missing == ["g2"]          # unknown ids are ignored
        assert verdict.refine_hint == "search replication lag"

    @pytest.mark.parametrize("response", [RuntimeError("judge down"), "not json", {"covered": []}])
    def test_llm_failure_falls_back_to_heuristic(self, response):
        pool, plan, cache = critic_inputs()
        trace = TraceRecorder()
        critic = Critic(CriticConfig(mode="llm"), llm_call=llm_from(ScriptedChatClient([response])),
                        embedding_cache=cache, trace=trace)
        verdict = critic.judge("q", plan, grounded_for("LAT001234", "NET000111"), pool, budget=BudgetTracker())
        assert verdict.fallback and verdict.mode == "heuristic" and verdict.missing == ["g2"]
        assert trace.events_named("critic_fallback")

    def test_both_mode_llm_sufficient_heuristic_insufficient_is_insufficient(self):
        pool, plan, cache = critic_inputs()
        client = ScriptedChatClient([{"sufficient": True, "covered": ["g1", "g2"], "missing": []}])
        verdict = Critic(CriticConfig(mode="both"), llm_call=llm_from(client), embedding_cache=cache).judge(
            "q", plan, grounded_for("LAT001234", "NET000111"), pool, budget=BudgetTracker())
        assert verdict.mode == "both" and not verdict.sufficient and verdict.missing == ["g2"]

    def test_both_mode_sufficient_when_both_agree(self):
        pool, plan, cache = critic_inputs()
        cache["replication"] = [0.0, 0.0, 1.0]
        client = ScriptedChatClient([{"sufficient": True, "covered": ["g1", "g2"], "missing": []}])
        verdict = Critic(CriticConfig(mode="both"), llm_call=llm_from(client), embedding_cache=cache).judge(
            "q", plan, grounded_for("LAT001234", "NET000111"), pool, budget=BudgetTracker())
        assert verdict.sufficient


# -- 6.x Self-Review -----------------------------------------------------------

class TestSelfReview:
    review = SelfReview(SelfReviewConfig(min_citations=1, min_cited_sentence_ratio=0.5), make_guardrails())

    def test_fabricated_citation_fails(self):
        verdict = self.review.check("The index was rebuilt last night. [CITATION:FAKE000001]", grounded_for("LAT001234"))
        assert not verdict.passed and "FAKE000001" in verdict.reasons[0]

    def test_citation_from_a_previous_turn_only_fails(self):
        # LAT002345 was cited in an earlier turn but is not grounded in this one
        verdict = self.review.check("Replication lag was fixed by tuning WAL. [CITATION:LAT002345]",
                                    grounded_for("LAT001234"))
        assert not verdict.passed and "LAT002345" in verdict.reasons[0]

    def test_under_cited_draft_fails(self):
        draft = ("The composite index fixed the latency [CITATION:LAT001234]. The pool size was doubled to "
                 "one hundred. Caching was added with Redis for reads. The slow query threshold was lowered.")
        verdict = self.review.check(draft, grounded_for("LAT001234"))
        assert not verdict.passed and "sentences are cited" in verdict.reasons[0]
        no_citations = self.review.check("The composite index fixed the latency issue quickly.", grounded_for("LAT001234"))
        assert any("at least 1" in r for r in no_citations.reasons)

    def test_fully_cited_draft_passes(self):
        draft = ("The ticket is resolved and its SLA was 240 minutes. [CITATION:LAT001234]\n"
                 "- Root cause was a missing composite index [CITATION:LAT001234]")
        assert self.review.check(draft, grounded_for("LAT001234")).passed

    def test_refusal_passes(self):
        assert self.review.check(STANDARD_REFUSAL, grounded_for()).passed

    def test_cited_sentence_ratio(self):
        assert cited_sentence_ratio("One two three four. [CITATION:A] Five six seven eight.") == 0.5
        assert cited_sentence_ratio("Short. Yes.") == 1.0

    def test_llm_check_can_fail_a_passing_draft(self):
        client = ScriptedChatClient([{"unsupported_claims": ["the fix took two days"]}])
        review = SelfReview(SelfReviewConfig(mode="llm"), make_guardrails(), llm_call=llm_from(client))
        verdict = review.check("Fixed by an index; it took two days. [CITATION:LAT001234]", grounded_for("LAT001234"),
                               budget=BudgetTracker())
        assert not verdict.passed and verdict.mode == "llm"
        assert "two days" in verdict.reasons[0]

    def test_llm_check_is_never_called_for_a_failing_draft(self):
        client = ScriptedChatClient([{"unsupported_claims": []}])
        review = SelfReview(SelfReviewConfig(mode="llm"), make_guardrails(), llm_call=llm_from(client))
        verdict = review.check("Fixed by magic. [CITATION:FAKE000001]", grounded_for("LAT001234"), budget=BudgetTracker())
        assert not verdict.passed and client.requests == []

    def test_llm_check_error_keeps_deterministic_pass(self):
        client = ScriptedChatClient([RuntimeError("down")])
        review = SelfReview(SelfReviewConfig(mode="llm"), make_guardrails(), llm_call=llm_from(client))
        assert review.check("Fixed by an index on user_id. [CITATION:LAT001234]", grounded_for("LAT001234"),
                            budget=BudgetTracker()).passed
