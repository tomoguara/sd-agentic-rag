"""Agentic RAG: the last four stages (Retrieve -> Re-rank -> Ground -> Generate)
run inside a bounded loop that a Planner controls, a Critic and a Self-Review
keep honest, Memory feeds context, and a Guard wraps.

Public API: AgenticRAGLoop, AgentConfig, TurnResult.
"""

from sd_agentic_rag.agent.config import AgentConfig, AgentConfigError
from sd_agentic_rag.agent.loop import AgenticRAGLoop
from sd_agentic_rag.agent.state import TurnResult

__all__ = ["AgenticRAGLoop", "AgentConfig", "AgentConfigError", "TurnResult"]
