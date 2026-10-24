"""Grounding across sub-goals and iterations (design D5).

Evidence accumulates in an EvidencePool for the whole turn. Each iteration
grounds the pool per sub-goal with the configured ContextGrounding (so every
sub-goal keeps up to `ground.max_passages` passages), merges the results by
passage, then applies `truncate_context` to `ground.max_tokens`. With one
sub-goal this is exactly the base path's single `ground_context` call. The
merged citation list is the turn's allow-list.
"""

from typing import Dict, List

from sd_agentic_rag.agent.state import Evidence
from sd_agentic_rag.sd_agentic_ground import ContextGrounding, GroundedContext


_SEPARATOR = "\n---\n"   # ContextGrounding's passage separator


class EvidencePool:
    """All evidence gathered in a turn, grouped by sub-goal and keyed by point id."""

    def __init__(self):
        self._by_goal: Dict[str, Dict[str, Evidence]] = {}

    def add(self, sub_goal_id: str, evidence: List[Evidence]) -> int:
        """Add evidence for a sub-goal; returns how many passages were new to it."""
        bucket = self._by_goal.setdefault(sub_goal_id, {})
        new = sum(1 for e in evidence if e.point_id not in bucket)
        for e in evidence:
            bucket[e.point_id] = e
        return new

    @property
    def sub_goal_ids(self) -> List[str]:
        return list(self._by_goal)

    def for_goal(self, sub_goal_id: str) -> List[Evidence]:
        return list(self._by_goal.get(sub_goal_id, {}).values())

    def all(self) -> List[Evidence]:
        """Every distinct passage in the pool (first sub-goal wins on duplicates)."""
        seen: Dict[str, Evidence] = {}
        for bucket in self._by_goal.values():
            for point_id, e in bucket.items():
                seen.setdefault(point_id, e)
        return list(seen.values())

    def by_ticket(self, ticket_id: str) -> List[Evidence]:
        return [e for e in self.all() if e.ticket_id == ticket_id]

    def __len__(self) -> int:
        return len(self.all())


def _split(context: GroundedContext):
    """(citation, passage block) pairs of a grounded context, or None if they cannot be aligned."""
    if not context.citations:
        return []
    parts = context.context_text.split(_SEPARATOR)
    if len(parts) != len(context.citations):
        return None
    return list(zip(context.citations, parts))


def merge_grounded(contexts: List[GroundedContext]) -> GroundedContext:
    """Merge per-sub-goal grounded contexts, dropping passages already included."""
    if len(contexts) == 1:
        return contexts[0]
    citations, parts, seen = [], [], set()
    for context in contexts:
        pairs = _split(context)
        if pairs is None:
            # Passage text contained the separator; keep the block whole
            pairs = [(c, context.context_text if i == 0 else "") for i, c in enumerate(context.citations)]
        for citation, part in pairs:
            key = (citation["ticket_id"], part)
            if key in seen:
                continue
            seen.add(key)
            citation = dict(citation, rank=len(citations) + 1)
            citations.append(citation)
            if part:
                parts.append(part)
    context_text = _SEPARATOR.join(parts)
    return GroundedContext(
        context_text=context_text,
        citations=citations,
        relevance_scores=[c["score"] for c in citations],
        total_tokens=len(context_text) // 4,
    )


def ground_evidence(grounder: ContextGrounding, evidence: List[Evidence], query: str) -> GroundedContext:
    """Ground one sub-goal's evidence with the unmodified ContextGrounding."""
    return grounder.ground_context(reranked_results=[e.to_ground_result() for e in evidence], query=query)


def ground_pool(grounder: ContextGrounding, pool: EvidencePool, queries: Dict[str, str]) -> GroundedContext:
    """Ground the pool per sub-goal, merge, and truncate to the token budget.

    Args:
        grounder: The pipeline's ContextGrounding
        pool: All evidence gathered this turn
        queries: sub-goal id -> query (passed to ground_context as `query`)
    """
    contexts = [
        ground_evidence(grounder, pool.for_goal(goal_id), queries.get(goal_id, ""))
        for goal_id in pool.sub_goal_ids
    ]
    contexts = [c for c in contexts if c.citations]
    if not contexts:
        return GroundedContext(context_text="", citations=[], relevance_scores=[], total_tokens=0)
    return grounder.truncate_context(merge_grounded(contexts))
