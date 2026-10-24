"""Agent configuration: the `agent:` section of config/config.yaml plus CLI overrides.

Defaults follow design D6 (budget), D7 (critic), D8 (self-review), D10
(planner) and D12 (memory). Values are validated on load; an invalid value
raises AgentConfigError naming the key.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import yaml


PLANNER_MODES = ("llm", "heuristic")
CRITIC_MODES = ("llm", "heuristic", "both")
SELF_REVIEW_MODES = ("heuristic", "llm")

DEFAULT_CONFIG_PATH = "config/config.yaml"


class AgentConfigError(ValueError):
    """Raised when the agent configuration contains an invalid value."""


@dataclass
class PlannerConfig:
    mode: str = "llm"
    max_sub_goals: int = 4
    max_top_k: int = 20          # Cap when refine doubles a sub-goal's top_k


@dataclass
class CriticConfig:
    mode: str = "llm"
    min_similarity: float = 0.5  # Heuristic: sub-goal query vs passage cosine


@dataclass
class SelfReviewConfig:
    mode: str = "heuristic"      # "llm" adds a tighten-only groundedness check
    min_citations: int = 1
    min_cited_sentence_ratio: float = 0.5


@dataclass
class BudgetConfig:
    max_iterations: int = 3
    max_llm_calls: int = 12
    max_total_tokens: int = 20000
    max_wall_seconds: float = 120.0
    max_revisions: int = 1


@dataclass
class LongTermConfig:
    enabled: bool = True
    top_k: int = 3
    min_similarity: float = 0.75


@dataclass
class MemoryConfig:
    enabled: bool = True
    db_path: str = "data/sd_agentic_memory.sqlite"
    recent_turns: int = 3
    turn_chars: int = 1500       # Per-turn text limit in the Planner prompt
    long_term: LongTermConfig = field(default_factory=LongTermConfig)


@dataclass
class AgentConfig:
    """Configuration for the agentic RAG loop."""

    top_k: int = 5
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    self_review: SelfReviewConfig = field(default_factory=SelfReviewConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AgentConfig":
        """Build a config from an `agent:` mapping; missing keys keep their defaults."""
        data = data or {}
        if not isinstance(data, dict):
            raise AgentConfigError("agent: must be a mapping")

        def mapping(name: str, values: Any, klass) -> Dict[str, Any]:
            values = values or {}
            if not isinstance(values, dict):
                raise AgentConfigError(f"agent.{name} must be a mapping")
            unknown = set(values) - set(klass.__dataclass_fields__)
            if unknown:
                raise AgentConfigError(f"Unknown key(s) in agent.{name}: {', '.join(sorted(unknown))}")
            return dict(values)

        top = mapping("", data, cls) if data else {}
        memory = mapping("memory", top.get("memory"), MemoryConfig)
        memory["long_term"] = LongTermConfig(
            **mapping("memory.long_term", memory.get("long_term"), LongTermConfig)
        )

        config = cls(
            top_k=top.get("top_k", 5),
            planner=PlannerConfig(**mapping("planner", top.get("planner"), PlannerConfig)),
            critic=CriticConfig(**mapping("critic", top.get("critic"), CriticConfig)),
            self_review=SelfReviewConfig(**mapping("self_review", top.get("self_review"), SelfReviewConfig)),
            budget=BudgetConfig(**mapping("budget", top.get("budget"), BudgetConfig)),
            memory=MemoryConfig(**memory),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, config_path: Optional[str] = None, **overrides) -> "AgentConfig":
        """Load the `agent:` section from config.yaml and apply CLI overrides.

        A missing file or section yields the defaults.
        """
        path = config_path or DEFAULT_CONFIG_PATH
        try:
            with open(path, "r") as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raw = {}
        return cls.from_dict(raw.get("agent")).with_overrides(**overrides)

    def with_overrides(
        self,
        max_iterations: Optional[int] = None,
        top_k: Optional[int] = None,
        no_memory: bool = False,
        db_path: Optional[str] = None,
    ) -> "AgentConfig":
        """Return a copy with CLI overrides applied (None leaves a value unchanged)."""
        config = AgentConfig.from_dict(self.to_dict())
        if max_iterations is not None:
            config.budget.max_iterations = max_iterations
        if top_k is not None:
            config.top_k = top_k
        if no_memory:
            config.memory.enabled = False
        if db_path:
            config.memory.db_path = db_path
        config.validate()
        return config

    def validate(self) -> None:
        """Raise AgentConfigError for the first invalid value."""
        def positive_int(name: str, value: Any, minimum: int = 1):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise AgentConfigError(f"agent.{name} must be an integer >= {minimum}, got {value!r}")

        def number(name: str, value: Any, low: float, high: Optional[float] = None, low_inclusive: bool = True):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AgentConfigError(f"agent.{name} must be a number, got {value!r}")
            too_low = value < low if low_inclusive else value <= low
            if too_low or (high is not None and value > high):
                bounds = f"in [{low}, {high}]" if high is not None else f"> {low}"
                raise AgentConfigError(f"agent.{name} must be {bounds}, got {value!r}")

        def choice(name: str, value: Any, allowed):
            if value not in allowed:
                raise AgentConfigError(f"agent.{name} must be one of {', '.join(allowed)}, got {value!r}")

        positive_int("top_k", self.top_k)
        choice("planner.mode", self.planner.mode, PLANNER_MODES)
        positive_int("planner.max_sub_goals", self.planner.max_sub_goals)
        positive_int("planner.max_top_k", self.planner.max_top_k)
        choice("critic.mode", self.critic.mode, CRITIC_MODES)
        number("critic.min_similarity", self.critic.min_similarity, -1.0, 1.0)
        choice("self_review.mode", self.self_review.mode, SELF_REVIEW_MODES)
        positive_int("self_review.min_citations", self.self_review.min_citations, minimum=0)
        number("self_review.min_cited_sentence_ratio", self.self_review.min_cited_sentence_ratio, 0.0, 1.0)
        positive_int("budget.max_iterations", self.budget.max_iterations)
        positive_int("budget.max_llm_calls", self.budget.max_llm_calls)
        positive_int("budget.max_total_tokens", self.budget.max_total_tokens)
        number("budget.max_wall_seconds", self.budget.max_wall_seconds, 0.0, low_inclusive=False)
        positive_int("budget.max_revisions", self.budget.max_revisions, minimum=0)
        if not isinstance(self.memory.db_path, str) or not self.memory.db_path:
            raise AgentConfigError("agent.memory.db_path must be a non-empty path")
        positive_int("memory.recent_turns", self.memory.recent_turns, minimum=0)
        positive_int("memory.turn_chars", self.memory.turn_chars)
        positive_int("memory.long_term.top_k", self.memory.long_term.top_k)
        number("memory.long_term.min_similarity", self.memory.long_term.min_similarity, -1.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
