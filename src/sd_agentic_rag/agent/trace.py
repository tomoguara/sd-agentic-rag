"""Per-turn trace of loop steps (design D11).

Each event is appended to an in-memory list (the `trace` in `--json`, printed
by `--verbose`) and, when memory is enabled, to the append-only audit through
the sink. Payloads carry ticket IDs, verdicts, counts and short redacted
previews - never full evidence text. Component calls are wrapped in
`trace_stage("agent.<component>")` spans with their latency recorded in a
MetricsCollector.
"""

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional

from sd_agentic_rag.sd_agentic_guard import Guardrails
from sd_agentic_rag.sd_agentic_observability import trace_stage


DEFAULT_PREVIEW_CHARS = 160

Sink = Callable[[Dict[str, Any]], None]


class TraceRecorder:
    """Records loop steps in order and forwards each one to an optional sink."""

    def __init__(
        self,
        sink: Optional[Sink] = None,
        metrics: Optional[Any] = None,
        tracer: Optional[Any] = None,
        preview_chars: int = DEFAULT_PREVIEW_CHARS,
        redact: Optional[Callable[[str], str]] = None,
    ):
        self.sink = sink
        self.metrics = metrics
        self.tracer = tracer
        self.preview_chars = preview_chars
        self.redact = redact or Guardrails().redact_pii
        self.events: List[Dict[str, Any]] = []
        self.iteration = 0

    def set_iteration(self, iteration: int) -> None:
        self.iteration = iteration

    def _clean(self, value: Any) -> Any:
        """Redact and truncate every string in a payload."""
        if isinstance(value, str):
            text = self.redact(value)
            if len(text) > self.preview_chars:
                text = text[:self.preview_chars] + "..."
            return text
        if isinstance(value, dict):
            return {str(k): self._clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._clean(v) for v in value]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return self._clean(str(value))

    def record(self, component: str, event: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Append one step and pass it to the sink."""
        entry = {
            "seq": len(self.events) + 1,
            "iteration": self.iteration,
            "component": component,
            "event": event,
            "payload": self._clean(payload or {}),
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.events.append(entry)
        if self.sink:
            self.sink(entry)
        return entry

    @contextmanager
    def span(self, component: str, extra: Optional[Dict] = None) -> Iterator[None]:
        """Trace a component call as `agent.<component>` and record its latency."""
        stage = f"agent.{component}"
        start = time.perf_counter()
        try:
            with trace_stage(stage, extra, self.tracer):
                yield
        finally:
            if self.metrics is not None:
                self.metrics.record_latency(stage, (time.perf_counter() - start) * 1000)

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def events_named(self, event: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e["event"] == event]
