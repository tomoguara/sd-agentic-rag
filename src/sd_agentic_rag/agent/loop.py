"""AgenticRAGLoop: the bounded answering loop (design D3).

    Planner -> Tool -> Re-rank -> Ground -> Critic -> Generate -> Self-Review,
    wrapped by the Guard, fed by Memory.

The loop receives an already-constructed SdAgenticRAGPipeline and reuses its
stage objects (one index, one BM25 build, one embedding client per process).
Control flow is plain code; the LLM is consulted only inside the Planner,
the Critic and (optionally) Self-Review, and every LLM-driven decision has a
deterministic fallback. Drafts are generated only from evidence grounded in
the current turn, and only that evidence is citable.
"""

import time
from typing import Any, Callable, Dict, List, Optional

from sd_agentic_rag.agent import prompts
from sd_agentic_rag.agent.budget import BudgetExceeded, BudgetTracker
from sd_agentic_rag.agent.config import AgentConfig
from sd_agentic_rag.agent.critic import Critic, store_vector_fetcher
from sd_agentic_rag.agent.grounding import EvidencePool, ground_pool
from sd_agentic_rag.agent.guard import AgentGuard
from sd_agentic_rag.agent.llm import json_call
from sd_agentic_rag.agent.memory import AgentMemory
from sd_agentic_rag.agent.planner import Planner, PlannerContext
from sd_agentic_rag.agent.self_review import SelfReview
from sd_agentic_rag.agent.state import GateVerdict, IterationRecord, Plan, TurnResult
from sd_agentic_rag.agent.tools import TICKET_ID_RE, build_tool_registry
from sd_agentic_rag.agent.trace import TraceRecorder
from sd_agentic_rag.sd_agentic_generate import GROUNDED_SYSTEM_PROMPT
from sd_agentic_rag.sd_agentic_ground import GroundedContext


class AgenticRAGLoop:
    """Answers one turn at a time through the bounded loop."""

    def __init__(
        self,
        pipeline: Any,
        config: Optional[AgentConfig] = None,
        memory: Optional[AgentMemory] = None,
        thread_id: str = "default",
        llm_call: Optional[Callable[..., Dict[str, Any]]] = None,
        tools: Optional[Dict[str, Any]] = None,
        fetch_vectors: Optional[Callable] = None,
        known_values: Optional[Dict[str, List[str]]] = None,
        clock: Callable[[], float] = time.monotonic,
        metrics: Optional[Any] = None,
    ):
        self.pipeline = pipeline
        self.config = config or AgentConfig()
        self.memory = memory if self.config.memory.enabled else None
        self.thread_id = thread_id
        self.llm_call = llm_call or json_call
        self.embedding_cache: Dict[str, List[float]] = {}
        self.tools = tools if tools is not None else build_tool_registry(pipeline, self.embedding_cache)
        if fetch_vectors is None and getattr(pipeline, "vector_store", None) is not None:
            fetch_vectors = store_vector_fetcher(pipeline.vector_store)
        self.fetch_vectors = fetch_vectors
        if known_values is None:
            retriever = getattr(pipeline, "hybrid_retriever", None)
            known_values = retriever.known_metadata_values() if retriever is not None else {}
        self.known_values = known_values
        self.clock = clock
        observability = getattr(pipeline, "observability", None)
        self.metrics = metrics if metrics is not None else getattr(observability, "collector", None)

    # -- helpers ---------------------------------------------------------

    @property
    def embed_service(self):
        return getattr(self.pipeline, "embed_service", None)

    def _embed_query(self, text: str) -> List[float]:
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.embed_service.embed_query(text)
        return self.embedding_cache[text]

    def _load_context(self, question: str, trace: TraceRecorder):
        """Planner context from memory, plus the question vector for long-term memory."""
        context = PlannerContext(known_values=self.known_values)
        question_vector = None
        if self.memory is None:
            return context, question_vector
        cfg = self.config.memory
        context.recent_turns = self.memory.recent_turns(self.thread_id, cfg.recent_turns)
        if cfg.long_term.enabled and self.embed_service is not None:
            question_vector = self._embed_query(question)
            for match in self.memory.search_resolutions(
                question_vector, self.embed_service.model,
                top_k=cfg.long_term.top_k, min_similarity=cfg.long_term.min_similarity,
            ):
                for ticket_id in match["cited_ticket_ids"]:
                    if ticket_id not in context.hints:
                        context.hints.append(ticket_id)
        trace.record("memory", "context", {
            "recent_turns": len(context.recent_turns),
            "remembered_ticket_ids": context.remembered_ticket_ids(),
            "hints": context.hints,
        })
        return context, question_vector

    def _generate(self, question: str, grounded: GroundedContext, budget: BudgetTracker,
                  trace: TraceRecorder, feedback: Optional[List[str]] = None, reserved: bool = False) -> str:
        if not reserved:
            budget.check_llm_call()
        system_prompt = GROUNDED_SYSTEM_PROMPT
        if feedback:
            system_prompt += prompts.REVISION_NOTE.format(reasons="\n".join(f"- {r}" for r in feedback))
        with trace.span("generate"):
            try:
                answer = self.pipeline.generate_pipeline.generate_answer(
                    question=question,
                    context=grounded.context_text,
                    citations=grounded.citations,
                    system_prompt=system_prompt,
                )
            except Exception:
                budget.record_llm_call(0)
                raise
        budget.record_llm_call(getattr(answer, "total_tokens", 0))
        trace.record("generate", "draft", {
            "revision": bool(feedback), "reserved": reserved,
            "tokens": getattr(answer, "total_tokens", 0), "preview": answer.answer,
        })
        return answer.answer

    @staticmethod
    def _signature(goal) -> tuple:
        """What makes two tool calls return the same evidence (a lookup depends only on its ticket ID)."""
        if goal.tool == "lookup":
            return (goal.tool, (goal.ticket_id or "").upper())
        return (goal.tool, goal.query, tuple(sorted((goal.filters or {}).items())), goal.top_k)

    def _refine(self, planner, question, plan, verdict, context, budget, executed, trace) -> Plan:
        """Refine the plan; tool calls identical to ones already run are not repeated.

        If the Planner only repeats earlier calls, the heuristic broadening is
        tried instead; an empty open set then means nothing new can be tried.
        """
        refined = planner.refine(question, plan, verdict, context, budget=budget)
        fresh = [g.id for g in refined.open_sub_goals if self._signature(g) not in executed]
        if not fresh and refined.open_sub_goals and refined.planner == "llm":
            trace.record("planner", "refine_repeated", {"repeated": [g.id for g in refined.open_sub_goals]})
            refined = planner.heuristic.refine(plan, verdict)
            fresh = [g.id for g in refined.open_sub_goals if self._signature(g) not in executed]
        refined.open_ids = fresh
        return refined

    @staticmethod
    def _relevance_question(question: str, context: PlannerContext, records: List[IterationRecord]) -> Optional[str]:
        """D17: previous question + current one when the turn looked up a remembered ticket it does not name."""
        if not context.recent_turns:
            return None
        remembered = set(context.remembered_ticket_ids()) - set(TICKET_ID_RE.findall(question))
        looked_up = {c.get("ticket_id") for r in records for c in r.tool_calls if c.get("tool") == "lookup"}
        if not remembered & looked_up:
            return None
        return f"{context.recent_turns[-1]['question']} {question}"

    def _citations(self, answer: str, grounded: GroundedContext, pool: EvidencePool, guard: AgentGuard):
        """Grounded citations the answer cites, in order of first mention, with the tool that found them."""
        cited = []
        for ticket_id in guard.guardrails.extract_citations(answer):
            if ticket_id in cited:
                continue
            if any(c["ticket_id"] == ticket_id for c in grounded.citations):
                cited.append(ticket_id)
        result = []
        for ticket_id in cited:
            citation = dict(next(c for c in grounded.citations if c["ticket_id"] == ticket_id))
            tools = list(dict.fromkeys(e.tool for e in pool.by_ticket(ticket_id)))
            citation["tool"] = ", ".join(tools) or "unknown"
            result.append(citation)
        return result

    # -- the turn ----------------------------------------------------------

    def run_turn(self, question: str, new_conversation: bool = False) -> TurnResult:
        cfg = self.config
        budget = BudgetTracker(cfg.budget, clock=self.clock)
        guard = AgentGuard(self.pipeline.guard_pipeline, budget, embedding_service=self.embed_service)

        turn_no = None
        sink = None
        if self.memory is not None:
            if new_conversation:
                self.memory.new_conversation(self.thread_id)
            turn_no = self.memory.next_turn_no(self.thread_id)
            sink = self.memory.step_sink(self.thread_id, turn_no)
        trace = TraceRecorder(sink=sink, metrics=self.metrics, redact=guard.redact)

        llm_call = self.llm_call
        planner = Planner(cfg.planner, cfg.top_k, llm_call=llm_call, trace=trace,
                          memory_turn_chars=cfg.memory.turn_chars)
        critic = Critic(cfg.critic, llm_call=llm_call,
                        embed_query=self._embed_query if self.embed_service is not None else None,
                        fetch_vectors=self.fetch_vectors, embedding_cache=self.embedding_cache, trace=trace)
        self_review = SelfReview(cfg.self_review, guardrails=guard.guardrails, llm_call=llm_call, trace=trace)

        q = guard.redact(question)
        trace.record("guard", "question", {"question": q, "redacted": q != question})

        pool = EvidencePool()
        records: List[IterationRecord] = []
        grounded = GroundedContext(context_text="", citations=[], relevance_scores=[], total_tokens=0)
        final: Optional[Dict[str, Any]] = None
        final_grounded = grounded
        stop_reason = "answered"
        partial = False
        executed = set()                 # signatures of tool calls already run this turn
        context = PlannerContext(known_values=self.known_values)
        question_vector = None

        try:
            context, question_vector = self._load_context(q, trace)
            with trace.span("planner"):
                plan: Plan = planner.plan(q, context, budget=budget,
                                          force_heuristic=cfg.budget.max_iterations == 1)
            while final is None:
                iteration = budget.start_iteration()
                trace.set_iteration(iteration)
                record = IterationRecord(iteration=iteration, plan=plan.to_dict())
                records.append(record)

                for goal in plan.open_sub_goals:
                    budget.check()
                    with trace.span(f"tool.{goal.tool}"):
                        evidence = guard.redact_evidence(self.tools[goal.tool].run(goal, goal.top_k or cfg.top_k))
                    executed.add(self._signature(goal))
                    new = pool.add(goal.id, evidence)
                    call = {"sub_goal": goal.id, "tool": goal.tool, "query": goal.query,
                            "ticket_id": goal.ticket_id, "filters": goal.filters,
                            "evidence_ticket_ids": list(dict.fromkeys(e.ticket_id for e in evidence)),
                            "count": len(evidence), "new": new}
                    record.tool_calls.append(call)
                    trace.record("tool", "tool_call", call)

                with trace.span("ground"):
                    grounded = ground_pool(self.pipeline.ground_pipeline, pool,
                                           {g.id: g.query for g in plan.sub_goals})
                record.grounded_ticket_ids = [c["ticket_id"] for c in grounded.citations]
                trace.record("ground", "grounded", {"ticket_ids": record.grounded_ticket_ids,
                                                    "tokens": grounded.total_tokens})

                with trace.span("critic"):
                    verdict = critic.judge(q, plan, grounded, pool, budget=budget)
                record.critic = verdict.to_dict()
                if not verdict.sufficient:
                    if iteration >= cfg.budget.max_iterations:
                        raise BudgetExceeded("max_iterations")
                    with trace.span("planner"):
                        plan = self._refine(planner, q, plan, verdict, context, budget, executed, trace)
                    if not plan.open_sub_goals:
                        stop_reason = "no_refinement"
                        break
                    continue

                feedback = None
                review = None
                while True:
                    draft = self._generate(q, grounded, budget, trace, feedback)
                    if guard.is_refusal(draft) and iteration < cfg.budget.max_iterations:
                        # D16: the Critic said sufficient but the generator could not answer
                        review = GateVerdict(gate="critic", passed=False, mode="draft_refusal",
                                             missing=[g.id for g in plan.sub_goals],
                                             reasons=["the draft was a refusal although the Critic judged "
                                                      "the evidence sufficient"])
                        record.drafts.append({"revision": len(record.drafts), "refusal": True})
                        trace.record("generate", "draft_refusal", {"grounded_ticket_ids": record.grounded_ticket_ids})
                        break
                    with trace.span("self_review"):
                        review = self_review.check(draft, grounded, budget=budget)
                    record.drafts.append({"revision": len(record.drafts), "self_review": review.to_dict()})
                    if review.passed:
                        with trace.span("guard"):
                            final = guard.finalize(draft, q, grounded,
                                                   relevance_question=self._relevance_question(q, context, records))
                        final_grounded = grounded
                        break
                    if not budget.can_revise():
                        break
                    budget.record_revision()
                    feedback = review.reasons
                if final is None:
                    if iteration >= cfg.budget.max_iterations:
                        raise BudgetExceeded("max_iterations")
                    with trace.span("planner"):
                        plan = self._refine(planner, q, plan, review, context, budget, executed, trace)
                    if not plan.open_sub_goals:
                        stop_reason = "no_refinement"
                        break
        except BudgetExceeded as e:
            stop_reason = e.reason
            trace.record("guard", "budget_exceeded", {"reason": e.reason, **budget.summary()})

        if final is None:
            final, final_grounded, partial = self._exhaust(q, pool, plan_queries(records), budget, guard,
                                                           self_review, trace, stop_reason,
                                                           self._relevance_question(q, context, records))

        answer = final["modified_answer"]
        refused = guard.is_refusal(answer)
        if refused:
            partial = False
        citations = [] if refused else self._citations(answer, final_grounded, pool, guard)
        guard_info = {
            "allowed": final.get("allowed", True),
            "enforced": final.get("enforced", False),
            "replaced": final.get("replaced", False),
            "actions": final.get("actions", []),
        }
        trace.record("guard", "final", {**guard_info, "refused": refused, "partial": partial,
                                        "cited_ticket_ids": [c["ticket_id"] for c in citations]})

        loop = budget.summary()
        loop["stop_reason"] = stop_reason
        loop["fallbacks"] = [e["event"] for e in trace.events
                             if e["event"] in ("planner_fallback", "critic_fallback")]

        result = TurnResult(
            question=q,
            answer=answer,
            citations=citations,
            grounded=list(final_grounded.citations),
            partial=partial,
            refused=refused,
            stop_reason=stop_reason,
            guard=guard_info,
            loop=loop,
            iterations=records,
            trace=trace.to_list(),
            thread_id=self.thread_id if self.memory is not None else None,
            turn_no=turn_no,
        )
        if self.memory is not None:
            self._remember(result, question_vector)
        return result

    def _exhaust(self, question, pool, queries, budget, guard, self_review, trace, stop_reason,
                 relevance_question=None):
        """X1: one best-effort draft over everything grounded this turn, gated, else refusal."""
        empty = GroundedContext(context_text="", citations=[], relevance_scores=[], total_tokens=0)
        grounded = ground_pool(self.pipeline.ground_pipeline, pool, queries) if len(pool) else empty
        trace.record("guard", "exhaust", {"stop_reason": stop_reason,
                                          "grounded_ticket_ids": [c["ticket_id"] for c in grounded.citations]})
        if grounded.citations:
            budget.use_reserved()
            draft = self._generate(question, grounded, budget, trace, reserved=True)
            review = self_review.check(draft, grounded, budget=budget, allow_llm=False)
            if review.passed:
                with trace.span("guard"):
                    final = guard.finalize(draft, question, grounded, relevance_question=relevance_question)
                partial = not final.get("replaced", False) and not guard.is_refusal(final["modified_answer"])
                return final, grounded, partial
        with trace.span("guard"):
            final = guard.finalize(guard.refusal(), question, empty)
        return final, empty, False

    def _remember(self, result: TurnResult, question_vector: Optional[List[float]]) -> None:
        """Save the turn; store a past resolution when the answer passed every gate."""
        resolution = None
        lt = self.config.memory.long_term
        if (lt.enabled and question_vector is not None and result.guard["allowed"]
                and not result.refused and result.citations):
            resolution = {"vector": question_vector, "embedding_model": self.embed_service.model}
        self.memory.save_turn(
            self.thread_id, result.turn_no, result.question, result.answer,
            result.cited_ticket_ids, partial=result.partial, refused=result.refused, resolution=resolution,
        )


def plan_queries(records: List[IterationRecord]) -> Dict[str, str]:
    """sub-goal id -> query across every plan the turn used (latest wins)."""
    queries: Dict[str, str] = {}
    for record in records:
        for goal in record.plan.get("sub_goals", []):
            queries[goal["id"]] = goal["query"]
    return queries
