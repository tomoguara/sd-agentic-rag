"""Planner: decomposes the question into sub-goals and refines them (design D10).

`heuristic` is deterministic: a lookup per ticket ID, a filtered retrieve
when a known service / severity / ticket type value is named, otherwise one
hybrid retrieve of the question; memory references and past-resolution hints
add lookups. `llm` (default) asks the chat model for the plan and falls back
to the heuristic plan on any failure or invalid plan, recording
`planner_fallback`. With `max_iterations == 1` the heuristic plan is forced.
"""

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from sd_agentic_rag.agent import prompts
from sd_agentic_rag.agent.config import PlannerConfig
from sd_agentic_rag.agent.llm import LLMCallError
from sd_agentic_rag.agent.state import GateVerdict, Plan, SubGoal
from sd_agentic_rag.agent.tools import (
    ALLOWED_TOOLS,
    FILTER_FIELDS,
    TICKET_ID_RE,
    PlanValidationError,
    validate_plan,
)
from sd_agentic_rag.agent.trace import TraceRecorder


# Metadata fields the heuristic Planner turns into filters when the question names a value
HEURISTIC_FILTER_FIELDS = ("service", "severity", "ticket_type")

# Words that refer back to a ticket from the recent conversation
REFERRING_RE = re.compile(r"\b(it|its|that ticket|this ticket|this issue|that issue|same)\b", re.IGNORECASE)


@dataclass
class PlannerContext:
    """What the Planner knows besides the question."""

    recent_turns: List[Dict[str, Any]] = field(default_factory=list)   # question, answer, cited_ticket_ids
    hints: List[str] = field(default_factory=list)                     # ticket IDs from past resolutions
    known_values: Dict[str, List[str]] = field(default_factory=dict)

    def remembered_ticket_ids(self) -> List[str]:
        """Ticket IDs cited in recent turns, most recent turn first."""
        ids: List[str] = []
        for turn in reversed(self.recent_turns):
            for ticket_id in turn.get("cited_ticket_ids") or []:
                if ticket_id not in ids:
                    ids.append(ticket_id)
        return ids


def _value_in_question(value: str, question: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", question, re.IGNORECASE) is not None


class HeuristicPlanner:
    """Deterministic plan and refine rules."""

    def __init__(self, config: Optional[PlannerConfig] = None, top_k: int = 5):
        self.config = config or PlannerConfig()
        self.top_k = top_k

    def plan(self, question: str, context: Optional[PlannerContext] = None) -> Plan:
        context = context or PlannerContext()
        sub_goals: List[SubGoal] = []
        planned_ids: List[str] = []

        def lookup(ticket_id: str, source: str):
            if ticket_id in planned_ids:
                return
            planned_ids.append(ticket_id)
            sub_goals.append(SubGoal(
                id=f"g{len(sub_goals) + 1}", query=question, tool="lookup",
                ticket_id=ticket_id, top_k=self.top_k, source=source,
            ))

        for ticket_id in TICKET_ID_RE.findall(question):
            lookup(ticket_id, "question")
        named_ids = bool(sub_goals)

        referred = bool(REFERRING_RE.search(question)) and not named_ids
        if referred:
            for ticket_id in context.remembered_ticket_ids():
                lookup(ticket_id, "memory")

        filters = {}
        for field_name in HEURISTIC_FILTER_FIELDS:
            for value in context.known_values.get(field_name, []):
                if _value_in_question(value, question):
                    filters[field_name] = value
                    break
        if filters:
            sub_goals.append(SubGoal(
                id=f"g{len(sub_goals) + 1}", query=question, tool="filtered_retrieve",
                filters=filters, top_k=self.top_k,
            ))
        elif not named_ids and not (referred and sub_goals):
            sub_goals.append(SubGoal(id=f"g{len(sub_goals) + 1}", query=question, tool="retrieve", top_k=self.top_k))

        for ticket_id in context.hints:
            lookup(ticket_id, "hint")

        return Plan(sub_goals=sub_goals[:self.config.max_sub_goals], planner="heuristic")

    def refine(self, plan: Plan, verdict: Optional[GateVerdict] = None) -> Plan:
        """Broaden the missing sub-goals.

        Retrieves drop their filters and double top_k (capped). A missing
        lookup becomes a retrieve of "<ticket_id> <query>", which finds tickets
        that reference the looked-up one, such as its follow-ups (D18). If
        nothing can be broadened the returned plan has no open sub-goals.
        """
        missing = set(verdict.missing) if verdict and verdict.missing else {g.id for g in plan.sub_goals}
        sub_goals, open_ids = [], []
        for goal in plan.sub_goals:
            if goal.id in missing:
                top_k = goal.top_k or self.top_k
                if goal.tool == "lookup":
                    query = goal.query if goal.ticket_id in goal.query else f"{goal.ticket_id} {goal.query}"
                    goal = replace(goal, tool="retrieve", query=query, ticket_id=None, top_k=top_k)
                    open_ids.append(goal.id)
                else:
                    broadened = replace(goal, tool="retrieve", filters={}, top_k=min(top_k * 2, self.config.max_top_k))
                    if broadened.top_k != goal.top_k or goal.filters:
                        open_ids.append(goal.id)
                        goal = broadened
            sub_goals.append(goal)
        return Plan(sub_goals=sub_goals, planner=plan.planner, open_ids=open_ids)


class Planner:
    """LLM planner with a deterministic fallback."""

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        top_k: int = 5,
        llm_call=None,
        trace: Optional[TraceRecorder] = None,
        memory_turn_chars: int = 1500,
    ):
        self.config = config or PlannerConfig()
        self.top_k = top_k
        self.heuristic = HeuristicPlanner(self.config, top_k)
        self.llm_call = llm_call
        self.trace = trace or TraceRecorder()
        self.memory_turn_chars = memory_turn_chars

    # -- prompt building -------------------------------------------------

    def _known_values_text(self, known_values: Dict[str, List[str]]) -> str:
        lines = []
        for field_name in FILTER_FIELDS:
            values = known_values.get(field_name) or []
            lines.append(f"  - {field_name}: {', '.join(values) if values else '(none known)'}")
        return "\n".join(lines)

    def _recent_turns_text(self, context: PlannerContext) -> str:
        if not context.recent_turns:
            return "(none)"
        limit = self.memory_turn_chars
        lines = []
        for turn in context.recent_turns:
            lines.append(f"Q: {str(turn.get('question', ''))[:limit]}")
            lines.append(f"A: {str(turn.get('answer', ''))[:limit]}")
            lines.append(f"Cited tickets: {', '.join(turn.get('cited_ticket_ids') or []) or '(none)'}")
        return "\n".join(lines)

    def _system_prompt(self, context: PlannerContext) -> str:
        return prompts.PLANNER_SYSTEM.format(
            max_sub_goals=self.config.max_sub_goals,
            filter_fields=", ".join(FILTER_FIELDS),
            known_values=self._known_values_text(context.known_values),
        )

    # -- parsing ---------------------------------------------------------

    def _canonical_filters(self, filters: Dict[str, Any], known: Dict[str, List[str]]) -> Dict[str, Any]:
        """Map filter values to the index's spelling when they match case-insensitively."""
        out = {}
        for field_name, value in filters.items():
            if isinstance(value, str):
                match = next((v for v in known.get(field_name, []) if v.lower() == value.strip().lower()), None)
                value = match or value.strip()
            out[field_name] = value
        return out

    def _parse_sub_goals(self, raw: Dict[str, Any], context: PlannerContext, question: str,
                         id_prefix: str = "g") -> List[SubGoal]:
        items = raw.get("sub_goals")
        if not isinstance(items, list) or not items:
            raise PlanValidationError("sub_goals must be a non-empty list")
        goals = []
        for i, item in enumerate(items, 1):
            if not isinstance(item, dict):
                raise PlanValidationError("each sub-goal must be an object")
            tool = item.get("tool")
            if tool not in ALLOWED_TOOLS:
                raise PlanValidationError(f"tool {tool!r} is not allowed")
            filters = item.get("filters") or {}
            if not isinstance(filters, dict):
                raise PlanValidationError("filters must be an object")
            ticket_id = item.get("ticket_id")
            if isinstance(ticket_id, str):
                ticket_id = ticket_id.strip().upper() or None
            query = str(item.get("query") or "").strip()
            if not query and tool == "lookup":
                # A lookup is keyed by its ticket ID; the question is a fine query
                query = question
            goals.append(SubGoal(
                id=str(item.get("id") or f"{id_prefix}{i}"),
                query=query,
                tool=tool,
                filters=self._canonical_filters(filters, context.known_values) if tool == "filtered_retrieve" else {},
                ticket_id=ticket_id if tool == "lookup" else None,
                top_k=self.top_k,
                source="llm",
            ))
        return goals

    def _cap(self, goals: List[SubGoal]) -> List[SubGoal]:
        if len(goals) > self.config.max_sub_goals:
            self.trace.record("planner", "planner_capped", {
                "proposed": len(goals), "max_sub_goals": self.config.max_sub_goals,
            })
        return goals[:self.config.max_sub_goals]

    # -- public API ------------------------------------------------------

    def plan(self, question: str, context: Optional[PlannerContext] = None, budget=None,
             force_heuristic: bool = False) -> Plan:
        """Plan the turn. BudgetExceeded propagates; any other failure falls back."""
        context = context or PlannerContext()
        if force_heuristic or self.config.mode == "heuristic" or self.llm_call is None:
            plan = self.heuristic.plan(question, context)
            self.trace.record("planner", "plan", plan.to_dict())
            return plan
        try:
            raw = self.llm_call(
                system=self._system_prompt(context),
                user=prompts.PLANNER_USER.format(
                    recent_turns=self._recent_turns_text(context),
                    hints=", ".join(context.hints) or "(none)",
                    question=question,
                ),
                schema={"sub_goals": list},
                budget=budget,
            )
            plan = validate_plan(Plan(sub_goals=self._cap(self._parse_sub_goals(raw, context, question)),
                                      planner="llm"))
        except (LLMCallError, PlanValidationError, ValueError, TypeError) as e:
            self.trace.record("planner", "planner_fallback", {"error": str(e)})
            plan = self.heuristic.plan(question, context)
        self.trace.record("planner", "plan", plan.to_dict())
        return plan

    def refine(self, question: str, plan: Plan, verdict: Optional[GateVerdict] = None,
               context: Optional[PlannerContext] = None, budget=None) -> Plan:
        """Refine after an insufficient Critic or failed Self-Review verdict."""
        context = context or PlannerContext()
        if self.config.mode == "heuristic" or self.llm_call is None or plan.planner == "heuristic":
            refined = self.heuristic.refine(plan, verdict)
            self.trace.record("planner", "refine", refined.to_dict())
            return refined
        feedback = {}
        if verdict is not None:
            feedback = {
                "gate": verdict.gate,
                "missing": verdict.missing,
                "refine_hint": verdict.refine_hint,
                "reasons": verdict.reasons,
            }
        try:
            raw = self.llm_call(
                system=self._system_prompt(context),
                user=prompts.PLANNER_REFINE_USER.format(
                    question=question,
                    plan=json.dumps([g.to_dict() for g in plan.sub_goals]),
                    feedback=json.dumps(feedback),
                ),
                schema={"sub_goals": list},
                budget=budget,
            )
            replacements = self._parse_sub_goals(raw, context, question, id_prefix="r")
            for goal in replacements:
                validate_plan(Plan(sub_goals=[goal]))
            by_id = {g.id: g for g in plan.sub_goals}
            open_ids = []
            for goal in replacements:
                by_id[goal.id] = goal
                if goal.id not in open_ids:
                    open_ids.append(goal.id)
            merged = list(by_id.values())
            if len(merged) > self.config.max_sub_goals:
                # Keep every replacement; drop the oldest closed sub-goals first
                closed = [g for g in merged if g.id not in open_ids]
                keep = self.config.max_sub_goals - len(open_ids)
                kept_closed = {g.id for g in closed[len(closed) - max(keep, 0):]} if keep > 0 else set()
                merged = [g for g in merged if g.id in open_ids or g.id in kept_closed]
                merged = self._cap(merged)
                open_ids = [i for i in open_ids if any(g.id == i for g in merged)]
            refined = validate_plan(Plan(sub_goals=merged, planner="llm", open_ids=open_ids))
        except (LLMCallError, PlanValidationError, ValueError, TypeError) as e:
            self.trace.record("planner", "planner_fallback", {"error": str(e), "stage": "refine"})
            refined = self.heuristic.refine(plan, verdict)
        self.trace.record("planner", "refine", refined.to_dict())
        return refined
