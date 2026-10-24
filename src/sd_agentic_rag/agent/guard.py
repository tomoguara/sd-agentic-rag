"""AgentGuard: the Guard wrapper around the whole loop (design D9).

It composes the base Guardrails rather than replacing them: PII redaction of
the question and of every evidence item before it reaches a prompt, the
trace or memory; and the same final-answer guard `answer` uses (enforcement
per `guard.enforce`, refusal handling, PII redaction) against the turn's
grounded allow-list. The turn's BudgetTracker is owned here.
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional

from sd_agentic_rag.agent.budget import BudgetTracker
from sd_agentic_rag.agent.state import Evidence
from sd_agentic_rag.sd_agentic_ground import GroundedContext
from sd_agentic_rag.sd_agentic_guard import Guardrails, STANDARD_REFUSAL


class AgentGuard:
    """Redacts what enters the loop and guards what leaves it."""

    def __init__(self, guardrails: Optional[Guardrails] = None, budget: Optional[BudgetTracker] = None,
                 embedding_service: Optional[Any] = None):
        self.guardrails = guardrails or Guardrails()
        self.budget = budget or BudgetTracker()
        self.embedding_service = embedding_service

    def redact(self, text: str) -> str:
        """Redact PII from text (always on: nothing unredacted may reach a prompt or memory)."""
        return self.guardrails.redact_pii(text or "")

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, list):
            return [self._redact_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._redact_value(v) for k, v in value.items()}
        return value

    def redact_evidence(self, items: List[Evidence]) -> List[Evidence]:
        """Copies of the evidence with PII redacted from text and metadata."""
        return [
            replace(e, text=self.redact(e.text), metadata=self._redact_value(e.metadata))
            for e in items
        ]

    def is_refusal(self, text: str) -> bool:
        return self.guardrails.check_refusal(text)

    def finalize(self, draft: str, question: str, grounded: GroundedContext,
                 relevance_question: Optional[str] = None) -> Dict[str, Any]:
        """Apply the base answer guardrails with the turn's grounded allow-list.

        Args:
            relevance_question: Question for the relevance gate when it differs
                from `question` (a follow-up gets the previous question too, D17);
                process_answer uses the question for nothing else

        Returns the Guardrails.process_answer result; `modified_answer` is the
        text to return (the standard refusal if enforcement replaced it).
        """
        return self.guardrails.process_answer(
            answer=draft,
            question=relevance_question or question,
            citations=grounded.citations,
            embedding_service=self.embedding_service,
        )

    @staticmethod
    def refusal() -> str:
        return STANDARD_REFUSAL
