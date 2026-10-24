"""Retrieval-family tools the Planner may call (design D4).

Exactly three tools exist: `retrieve`, `lookup` and `filtered_retrieve`.
Each returns ticket-attributed Evidence whose text starts with the ticket's
state line, so state facts (status, SLA, ...) are citable through the same
ticket ID. Evidence without a ticket ID is discarded. Nothing a tool returns
is citable until it passes through Ground.
"""

import re
from typing import Any, Dict, List, Optional, Protocol

from sd_agentic_rag.agent.state import Evidence, Plan, SubGoal
from sd_agentic_rag.sd_agentic_rag_pipeline import rank_candidates, retrieve_and_rank


ALLOWED_TOOLS = ("retrieve", "lookup", "filtered_retrieve")
FILTER_FIELDS = ("service", "severity", "ticket_type", "status")
TICKET_ID_RE = re.compile(r"\b[A-Z]{2,5}\d{6}\b")

# (payload field, label) rendered in the state line, in order
STATE_FIELDS = (
    ("status", "Status"),
    ("priority", "Priority"),
    ("severity", "Severity"),
    ("sla_minutes", "SLA"),
    ("assigned_to", "Assigned"),
)

# Ranking keys ContextGrounding sorts by; copied from the tool result unchanged
_SCORE_KEYS = ("rerank_score", "fused_score", "score")


class PlanValidationError(ValueError):
    """A plan names a tool or filter field outside the allowed set, or is malformed."""


def render_state_line(metadata: Dict[str, Any]) -> str:
    """e.g. 'Status: Resolved | Priority: P2 | SLA: 60 min'; missing fields are omitted."""
    parts = []
    for key, label in STATE_FIELDS:
        value = metadata.get(key)
        if value is None or value == "":
            continue
        if key == "sla_minutes":
            parts.append(f"{label}: {value} min")
        else:
            parts.append(f"{label}: {value}")
    return " | ".join(parts)


def evidence_from_result(result: Dict[str, Any], tool: str) -> Optional[Evidence]:
    """Convert a retriever/store result into Evidence; None if it has no ticket ID."""
    payload = result.get("payload") or {}
    ticket_id = payload.get("ticket_id") or result.get("ticket_id")
    if not ticket_id:
        return None
    text = payload.get("chunk_text") or result.get("document") or ""
    metadata = {k: v for k, v in payload.items() if k not in ("chunk_text", "ticket_id")}
    state_line = render_state_line(metadata)
    if state_line:
        text = f"{state_line}\n{text}"
    scores = {k: result[k] for k in _SCORE_KEYS if isinstance(result.get(k), (int, float))}
    return Evidence(
        point_id=str(result.get("id") or result.get("doc_id") or f"{ticket_id}:{payload.get('chunk_index', 0)}"),
        ticket_id=ticket_id,
        text=text,
        metadata=metadata,
        tool=tool,
        score=float(result.get("rerank_score", result.get("fused_score", result.get("score", 0.0))) or 0.0),
        retriever=result.get("retriever", tool),
        scores=scores,
    )


def _to_evidence(results: List[Dict[str, Any]], tool: str) -> List[Evidence]:
    evidence = (evidence_from_result(r, tool) for r in results)
    return [e for e in evidence if e is not None]


class Tool(Protocol):
    name: str

    def run(self, sub_goal: SubGoal, top_k: int) -> List[Evidence]:
        ...


class RetrieveTool:
    """Hybrid search for a sub-query - exactly the base `answer` path's retrieval."""

    name = "retrieve"

    def __init__(self, hybrid_retriever, rerank_pipeline=None, rerank_enabled: bool = False,
                 embedding_cache: Optional[Dict[str, List[float]]] = None):
        self.hybrid_retriever = hybrid_retriever
        self.rerank_pipeline = rerank_pipeline
        self.rerank_enabled = rerank_enabled
        self.embedding_cache = embedding_cache if embedding_cache is not None else {}

    def _embed(self, query: str) -> Optional[List[float]]:
        service = getattr(self.hybrid_retriever, "embedding_service", None)
        if service is None:
            return None
        if query not in self.embedding_cache:
            self.embedding_cache[query] = service.embed_query(query)
        return self.embedding_cache[query]

    def run(self, sub_goal: SubGoal, top_k: int) -> List[Evidence]:
        _, ranked = retrieve_and_rank(
            hybrid_retriever=self.hybrid_retriever,
            rerank_pipeline=self.rerank_pipeline,
            query=sub_goal.query,
            top_k=top_k,
            rerank_enabled=self.rerank_enabled,
            query_embedding=self._embed(sub_goal.query),
        )
        return _to_evidence(ranked, self.name)


class LookupTool:
    """All indexed chunks of one ticket, ordered by chunk_index; empty for an unknown ID.

    A lookup is an exact fetch, so its chunks are not re-ranked (a low
    re-rank score would otherwise drop the very ticket that was asked for);
    they still pass through Ground.
    """

    name = "lookup"

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def run(self, sub_goal: SubGoal, top_k: int) -> List[Evidence]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        ticket_id = (sub_goal.ticket_id or "").strip().upper()
        if not ticket_id:
            return []
        points, offset = [], None
        while True:
            batch, offset = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                scroll_filter=Filter(must=[FieldCondition(key="ticket_id", match=MatchValue(value=ticket_id))]),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if offset is None:
                break
        points.sort(key=lambda p: (p.payload or {}).get("chunk_index", 0))
        results = [
            {"id": str(p.id), "payload": p.payload, "score": 1.0, "retriever": "lookup"}
            for p in points
            if (p.payload or {}).get("ticket_id") == ticket_id
        ]
        return _to_evidence(results, self.name)


def _payload_matches(payload: Dict[str, Any], filters: Dict[str, str]) -> bool:
    return all((payload or {}).get(field) == value for field, value in filters.items())


class FilteredRetrieveTool:
    """Hybrid search restricted by metadata on both the vector and the keyword side."""

    name = "filtered_retrieve"

    def __init__(self, hybrid_retriever, rerank_pipeline=None, rerank_enabled: bool = False,
                 embedding_cache: Optional[Dict[str, List[float]]] = None):
        self.hybrid_retriever = hybrid_retriever
        self.rerank_pipeline = rerank_pipeline
        self.rerank_enabled = rerank_enabled
        self.embedding_cache = embedding_cache if embedding_cache is not None else {}

    def run(self, sub_goal: SubGoal, top_k: int) -> List[Evidence]:
        from qdrant_client.models import FieldCondition, MatchValue

        retriever = self.hybrid_retriever
        filters = dict(sub_goal.filters or {})
        validate_filters(filters)
        query = sub_goal.query

        if query not in self.embedding_cache:
            self.embedding_cache[query] = retriever.embedding_service.embed_query(query)
        conditions = [FieldCondition(key=f, match=MatchValue(value=v)) for f, v in filters.items()]
        vector_results = retriever.vector_store.search(
            query_vector=self.embedding_cache[query],
            top_k=retriever.vector_top_k,
            filter_conditions=conditions or None,
        )
        vector_results = [r for r in vector_results if _payload_matches(r.get("payload"), filters)]

        bm25_results = []
        if retriever.bm25_index:
            # Score the whole index, then keep the matching documents up to bm25_top_k
            everything = retriever.bm25_index.search(query, top_k=len(retriever.bm25_index.documents))
            bm25_results = [r for r in everything if _payload_matches(r.get("payload"), filters)]
            bm25_results = bm25_results[:retriever.bm25_top_k]

        fused = retriever.rrf_fusion(vector_results, bm25_results, k=retriever.k_rrf)
        ranked = rank_candidates(self.rerank_pipeline, query, fused, top_k, self.rerank_enabled)
        return _to_evidence(ranked, self.name)


def build_tool_registry(pipeline, embedding_cache: Optional[Dict[str, List[float]]] = None) -> Dict[str, Tool]:
    """The three tools over an already-constructed SdAgenticRAGPipeline's stage objects."""
    cache = embedding_cache if embedding_cache is not None else {}
    rerank_enabled = bool(getattr(pipeline, "rerank_enabled", False))
    return {
        "retrieve": RetrieveTool(pipeline.hybrid_retriever, pipeline.rerank_pipeline, rerank_enabled, cache),
        "lookup": LookupTool(pipeline.vector_store),
        "filtered_retrieve": FilteredRetrieveTool(pipeline.hybrid_retriever, pipeline.rerank_pipeline,
                                                  rerank_enabled, cache),
    }


def validate_filters(filters: Any) -> None:
    if not isinstance(filters, dict):
        raise PlanValidationError("filters must be an object")
    for field, value in filters.items():
        if field not in FILTER_FIELDS:
            raise PlanValidationError(f"filter field {field!r} is not allowed (allowed: {', '.join(FILTER_FIELDS)})")
        if not isinstance(value, str) or not value:
            raise PlanValidationError(f"filter {field!r} must be a non-empty string")


def validate_sub_goal(sub_goal: SubGoal) -> None:
    """Raise PlanValidationError unless the sub-goal uses an allowed tool correctly."""
    if sub_goal.tool not in ALLOWED_TOOLS:
        raise PlanValidationError(f"tool {sub_goal.tool!r} is not allowed (allowed: {', '.join(ALLOWED_TOOLS)})")
    if not isinstance(sub_goal.query, str) or not sub_goal.query.strip():
        raise PlanValidationError(f"sub-goal {sub_goal.id!r} has no query")
    validate_filters(sub_goal.filters or {})
    if sub_goal.tool == "lookup":
        if not sub_goal.ticket_id or not TICKET_ID_RE.fullmatch(sub_goal.ticket_id):
            raise PlanValidationError(f"lookup {sub_goal.id!r} needs a ticket_id like LAT001234")
    if sub_goal.tool == "filtered_retrieve" and not sub_goal.filters:
        raise PlanValidationError(f"filtered_retrieve {sub_goal.id!r} needs at least one filter")


def validate_plan(plan: Plan, max_sub_goals: Optional[int] = None) -> Plan:
    """Validate every sub-goal; raise PlanValidationError on the first problem."""
    if not plan.sub_goals:
        raise PlanValidationError("plan has no sub-goals")
    if max_sub_goals is not None and len(plan.sub_goals) > max_sub_goals:
        raise PlanValidationError(f"plan has {len(plan.sub_goals)} sub-goals (max {max_sub_goals})")
    ids = [g.id for g in plan.sub_goals]
    if len(set(ids)) != len(ids):
        raise PlanValidationError("sub-goal ids must be unique")
    for sub_goal in plan.sub_goals:
        validate_sub_goal(sub_goal)
    return plan
