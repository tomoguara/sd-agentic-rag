"""Hard per-turn budget (design D6), owned by the Guard.

Limits apply to the loop's own work. One generate call is reserved for the
exhaustion best-effort draft and may be used once even after a limit was hit,
so a turn makes at most `max_llm_calls + 1` LLM calls.
"""

import time
from typing import Callable, Dict, Optional

from sd_agentic_rag.agent.config import BudgetConfig


class BudgetExceeded(Exception):
    """A budget limit was reached; `reason` is the turn's stop reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class BudgetTracker:
    """Counts iterations, LLM calls, tokens, wall clock and revisions for one turn."""

    def __init__(self, config: Optional[BudgetConfig] = None, clock: Callable[[], float] = time.monotonic):
        self.config = config or BudgetConfig()
        self.clock = clock
        self.started_at = clock()
        self.iterations = 0
        self.llm_calls = 0
        self.total_tokens = 0
        self.revisions = 0               # Revisions in the current iteration
        self.total_revisions = 0
        self.reserved_used = False
        self.exceeded: Optional[str] = None

    @property
    def elapsed(self) -> float:
        return self.clock() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.config.max_wall_seconds - self.elapsed)

    def _exceed(self, reason: str):
        self.exceeded = self.exceeded or reason
        raise BudgetExceeded(reason)

    def check(self) -> None:
        """Raise if the wall clock or token limit is reached (before any tool or LLM call)."""
        if self.elapsed >= self.config.max_wall_seconds:
            self._exceed("max_wall_seconds")
        if self.total_tokens >= self.config.max_total_tokens:
            self._exceed("max_total_tokens")

    def start_iteration(self) -> int:
        """Begin the next iteration; raise once max_iterations have run."""
        if self.iterations >= self.config.max_iterations:
            self._exceed("max_iterations")
        self.check()
        self.iterations += 1
        self.revisions = 0
        return self.iterations

    def check_llm_call(self) -> None:
        """Raise if another (non-reserved) LLM call would exceed the budget."""
        if self.llm_calls >= self.config.max_llm_calls:
            self._exceed("max_llm_calls")
        self.check()

    def record_llm_call(self, tokens: int = 0) -> None:
        """Count an LLM call that was made and its token usage."""
        self.llm_calls += 1
        self.total_tokens += int(tokens or 0)

    def can_revise(self) -> bool:
        return self.revisions < self.config.max_revisions

    def record_revision(self) -> None:
        if not self.can_revise():
            self._exceed("max_revisions")
        self.revisions += 1
        self.total_revisions += 1

    def use_reserved(self) -> None:
        """Claim the reserved exhaustion call; usable exactly once per turn."""
        if self.reserved_used:
            raise BudgetExceeded("reserved_call_used")
        self.reserved_used = True

    def summary(self) -> Dict:
        return {
            "iterations": self.iterations,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "elapsed_s": round(self.elapsed, 3),
            "revisions": self.total_revisions,
            "reserved_call_used": self.reserved_used,
            "limits": {
                "max_iterations": self.config.max_iterations,
                "max_llm_calls": self.config.max_llm_calls,
                "max_total_tokens": self.config.max_total_tokens,
                "max_wall_seconds": self.config.max_wall_seconds,
                "max_revisions": self.config.max_revisions,
            },
        }
