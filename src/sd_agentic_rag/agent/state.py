"""Loop state: plans, evidence, gate verdicts and turn results.

Every dataclass has a JSON-serialisable `to_dict()`; that form is what the
trace, the `--json` output and the memory audit carry.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SubGoal:
    """One part of the question, answered by one tool call."""

    id: str
    query: str
    tool: str                                    # retrieve | lookup | filtered_retrieve
    filters: Dict[str, str] = field(default_factory=dict)
    ticket_id: Optional[str] = None              # lookup only
    top_k: Optional[int] = None                  # None = AgentConfig.top_k
    source: str = "question"                     # question | memory | hint | llm

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    """Sub-goals for the turn; `open_ids` are the ones the next iteration runs."""

    sub_goals: List[SubGoal]
    planner: str = "heuristic"                   # heuristic | llm
    open_ids: Optional[List[str]] = None         # None = all sub-goals are open

    @property
    def open_sub_goals(self) -> List[SubGoal]:
        if self.open_ids is None:
            return list(self.sub_goals)
        return [g for g in self.sub_goals if g.id in self.open_ids]

    def get(self, sub_goal_id: str) -> Optional[SubGoal]:
        return next((g for g in self.sub_goals if g.id == sub_goal_id), None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "planner": self.planner,
            "sub_goals": [g.to_dict() for g in self.sub_goals],
            "open_ids": [g.id for g in self.open_sub_goals],
        }


@dataclass
class Evidence:
    """A ticket-attributed passage returned by a tool.

    `text` starts with the ticket's state line (design D4). `vector` is the
    stored passage vector, fetched lazily by the heuristic Critic; it is never
    serialised.
    """

    point_id: str
    ticket_id: str
    text: str
    metadata: Dict[str, Any]
    tool: str
    score: float = 0.0
    retriever: str = "unknown"
    scores: Dict[str, float] = field(default_factory=dict)   # rerank_score / fused_score / score
    vector: Optional[List[float]] = None

    def to_ground_result(self) -> Dict[str, Any]:
        """The result shape ContextGrounding resolves (payload + ranking scores)."""
        payload = dict(self.metadata)
        payload["ticket_id"] = self.ticket_id
        payload["chunk_text"] = self.text
        result = {"id": self.point_id, "payload": payload, "retriever": self.retriever, "tool": self.tool}
        result.update(self.scores)
        return result

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("vector", None)
        return data


@dataclass
class GateVerdict:
    """Outcome of a gate: the Critic (sufficient?) or Self-Review (passed?)."""

    gate: str                                    # critic | self_review
    passed: bool
    mode: str = "heuristic"
    covered: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    refine_hint: str = ""
    fallback: bool = False

    @property
    def sufficient(self) -> bool:
        return self.passed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IterationRecord:
    """What one loop iteration did."""

    iteration: int
    plan: Dict[str, Any]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    grounded_ticket_ids: List[str] = field(default_factory=list)
    critic: Optional[Dict[str, Any]] = None
    drafts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TurnResult:
    """The answer to one turn plus everything needed to explain it."""

    question: str
    answer: str
    citations: List[Dict[str, Any]]              # Grounded citations the answer cites (verified)
    grounded: List[Dict[str, Any]]               # All grounded citations (the allow-list)
    partial: bool
    refused: bool
    stop_reason: Optional[str]
    guard: Dict[str, Any]
    loop: Dict[str, Any]
    iterations: List[IterationRecord] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    thread_id: Optional[str] = None
    turn_no: Optional[int] = None

    @property
    def cited_ticket_ids(self) -> List[str]:
        return [c["ticket_id"] for c in self.citations]

    @property
    def grounded_ticket_ids(self) -> List[str]:
        return [c["ticket_id"] for c in self.grounded]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "grounded": self.grounded,
            "partial": self.partial,
            "refused": self.refused,
            "stop_reason": self.stop_reason,
            "guard": self.guard,
            "loop": self.loop,
            "iterations": [r.to_dict() for r in self.iterations],
            "trace": self.trace,
            "thread_id": self.thread_id,
            "turn_no": self.turn_no,
        }
