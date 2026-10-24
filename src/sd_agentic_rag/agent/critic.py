"""Critic: decides before generation whether the grounded evidence suffices (design D7).

- Pre-check (all modes): nothing grounded -> insufficient, no LLM call.
- `heuristic`: every sub-goal needs a grounded passage whose stored vector has
  cosine >= `min_similarity` with the sub-goal's query embedding. A lookup
  sub-goal is covered when its ticket is grounded.
- `llm` (default): an LLM judge; any error or invalid verdict falls back to
  the heuristic verdict for that iteration (`critic_fallback`).
- `both`: sufficient only if both verdicts say so.

The Critic only returns a verdict; it cannot add or remove evidence.
"""

from typing import Callable, Dict, List, Optional

import numpy as np

from sd_agentic_rag.agent import prompts
from sd_agentic_rag.agent.config import CriticConfig
from sd_agentic_rag.agent.grounding import EvidencePool
from sd_agentic_rag.agent.llm import LLMCallError
from sd_agentic_rag.agent.state import GateVerdict, Plan
from sd_agentic_rag.agent.trace import TraceRecorder
from sd_agentic_rag.sd_agentic_ground import GroundedContext


VectorFetcher = Callable[[List[str]], Dict[str, List[float]]]


def store_vector_fetcher(vector_store) -> VectorFetcher:
    """Fetch stored passage vectors by point id (no embedding calls)."""
    def fetch(point_ids: List[str]) -> Dict[str, List[float]]:
        if not point_ids:
            return {}
        points = vector_store.client.retrieve(
            collection_name=vector_store.collection_name,
            ids=point_ids,
            with_payload=False,
            with_vectors=True,
        )
        return {str(p.id): p.vector for p in points if p.vector is not None}
    return fetch


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / denom if denom else 0.0


class Critic:
    """Evidence-sufficiency gate."""

    def __init__(
        self,
        config: Optional[CriticConfig] = None,
        llm_call=None,
        embed_query: Optional[Callable[[str], List[float]]] = None,
        fetch_vectors: Optional[VectorFetcher] = None,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
        trace: Optional[TraceRecorder] = None,
    ):
        self.config = config or CriticConfig()
        self.llm_call = llm_call
        self.embed_query = embed_query
        self.fetch_vectors = fetch_vectors
        self.embedding_cache = embedding_cache if embedding_cache is not None else {}
        self.trace = trace or TraceRecorder()

    def _query_vector(self, query: str) -> Optional[List[float]]:
        if query not in self.embedding_cache and self.embed_query is not None:
            self.embedding_cache[query] = self.embed_query(query)
        return self.embedding_cache.get(query)

    def heuristic(self, plan: Plan, grounded: GroundedContext, pool: EvidencePool) -> GateVerdict:
        grounded_ids = {c["ticket_id"] for c in grounded.citations}
        passages = [e for e in pool.all() if e.ticket_id in grounded_ids]
        missing_vectors = [e.point_id for e in passages if e.vector is None]
        if missing_vectors and self.fetch_vectors is not None:
            vectors = self.fetch_vectors(missing_vectors)
            for e in passages:
                if e.vector is None and e.point_id in vectors:
                    e.vector = vectors[e.point_id]

        covered, missing, reasons = [], [], []
        for goal in plan.sub_goals:
            if goal.tool == "lookup":
                ok = goal.ticket_id in grounded_ids
                reason = f"{goal.id}: ticket {goal.ticket_id} {'grounded' if ok else 'not grounded'}"
            else:
                query_vector = self._query_vector(goal.query)
                best = max(
                    (cosine(query_vector, e.vector) for e in passages if e.vector is not None),
                    default=0.0,
                ) if query_vector is not None else 0.0
                ok = best >= self.config.min_similarity
                reason = f"{goal.id}: best similarity {best:.3f} (min {self.config.min_similarity})"
            (covered if ok else missing).append(goal.id)
            reasons.append(reason)
        return GateVerdict(gate="critic", passed=not missing, mode="heuristic",
                           covered=covered, missing=missing, reasons=reasons)

    def _llm(self, question: str, plan: Plan, grounded: GroundedContext, budget) -> GateVerdict:
        raw = self.llm_call(
            system=prompts.CRITIC_SYSTEM,
            user=prompts.CRITIC_USER.format(
                question=question,
                sub_goals="\n".join(f"- {g.id}: {g.query}" + (f" (ticket {g.ticket_id})" if g.ticket_id else "")
                                    for g in plan.sub_goals),
                context=grounded.context_text,
            ),
            schema={"sufficient": bool},
            budget=budget,
        )
        ids = [g.id for g in plan.sub_goals]
        for key in ("covered", "missing"):
            if key in raw and not isinstance(raw[key], list):
                raise LLMCallError(f"{key} must be a list")
        # Only ids of this plan's sub-goals count; anything else the judge names is ignored
        covered = [str(i) for i in raw.get("covered") or [] if str(i) in ids]
        missing = [str(i) for i in raw.get("missing") or [] if str(i) in ids]
        sufficient = bool(raw["sufficient"])
        if not sufficient and not missing:
            missing = [i for i in ids if i not in covered] or ids
        if sufficient:
            missing = []
        hint = raw.get("refine_hint")
        return GateVerdict(gate="critic", passed=sufficient, mode="llm", covered=covered, missing=missing,
                           refine_hint=hint if isinstance(hint, str) else "")

    def judge(self, question: str, plan: Plan, grounded: GroundedContext, pool: EvidencePool,
              budget=None) -> GateVerdict:
        """Verdict for the current grounded evidence. BudgetExceeded propagates."""
        if not grounded.citations:
            verdict = GateVerdict(gate="critic", passed=False, mode="precheck",
                                  missing=[g.id for g in plan.sub_goals], reasons=["no passage grounded"])
            self.trace.record("critic", "verdict", verdict.to_dict())
            return verdict

        mode = self.config.mode
        if mode == "heuristic" or self.llm_call is None:
            verdict = self.heuristic(plan, grounded, pool)
        else:
            try:
                llm_verdict = self._llm(question, plan, grounded, budget)
            except LLMCallError as e:
                verdict = self.heuristic(plan, grounded, pool)
                verdict.fallback = True
                self.trace.record("critic", "critic_fallback", {"error": str(e)})
            else:
                if mode == "both":
                    heuristic = self.heuristic(plan, grounded, pool)
                    missing = list(dict.fromkeys(llm_verdict.missing + heuristic.missing))
                    verdict = GateVerdict(
                        gate="critic", passed=llm_verdict.passed and heuristic.passed, mode="both",
                        covered=[i for i in llm_verdict.covered if i not in missing],
                        missing=missing, reasons=heuristic.reasons, refine_hint=llm_verdict.refine_hint,
                    )
                else:
                    verdict = llm_verdict
        self.trace.record("critic", "verdict", verdict.to_dict())
        return verdict
