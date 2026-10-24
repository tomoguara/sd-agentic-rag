"""RAG Agent Command - Answer through the bounded agentic loop.

This module implements the agent command for the SdAgentic RAG CLI. Modes:
a single turn (--prompt), interactive chat (--chat), the thread's current
conversation (--history) and its step audit (--audit [--full]).
"""

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


EXIT_WORDS = ("exit", "quit")
AUDIT_PREVIEW_CHARS = 120


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


class RagAgentCommand:
    """Command to answer support questions through the agentic loop."""

    def __init__(self, verbose: bool = False, debug: bool = False,
                 pipeline_factory: Optional[Callable[[], Any]] = None,
                 input_stream: Optional[TextIO] = None, output_stream: Optional[TextIO] = None,
                 llm_call: Optional[Callable[..., Dict[str, Any]]] = None):
        self.verbose = verbose or debug
        self.debug = debug
        self.pipeline_factory = pipeline_factory
        self.llm_call = llm_call
        self.input_stream = input_stream
        self.output_stream = output_stream

    # -- setup -------------------------------------------------------------

    def _pipeline(self):
        """Build the base pipeline; its diagnostics go to stderr so --json stays parseable."""
        with contextlib.redirect_stdout(sys.stderr):
            if self.pipeline_factory:
                pipeline = self.pipeline_factory()
            else:
                from sd_agentic_rag import SdAgenticRAGPipeline
                pipeline = SdAgenticRAGPipeline()
        if not getattr(pipeline, "store_enabled", False) or getattr(pipeline, "hybrid_retriever", None) is None:
            raise RuntimeError("Vector store is not available - is Qdrant running? (scripts/qdrant-db.sh --start)")
        if getattr(pipeline, "generate_pipeline", None) is None or getattr(pipeline, "ground_pipeline", None) is None:
            raise RuntimeError("Ground and Generate stages are not available")
        return pipeline

    def _config(self, config_path, max_iterations, top_k, no_memory, db):
        from sd_agentic_rag.agent import AgentConfig
        from sd_agentic_rag.agent.memory import resolve_db_path
        config = AgentConfig.load(config_path, max_iterations=max_iterations, top_k=top_k, no_memory=no_memory)
        config.memory.db_path = resolve_db_path(db, config.memory.db_path)
        return config

    def _loop(self, config, thread_id):
        from sd_agentic_rag.agent import AgenticRAGLoop
        from sd_agentic_rag.agent.memory import AgentMemory
        pipeline = self._pipeline()
        memory = AgentMemory(config.memory.db_path) if config.memory.enabled else None
        return AgenticRAGLoop(pipeline, config, memory=memory, thread_id=thread_id, llm_call=self.llm_call)

    # -- run ---------------------------------------------------------------

    def run(self, mode: str, prompt: Optional[str] = None, thread_id: str = "default", new: bool = False,
            max_iterations: Optional[int] = None, top_k: Optional[int] = None, no_memory: bool = False,
            db: Optional[str] = None, config_path: Optional[str] = None, full: bool = False,
            output_format: str = "markdown") -> Dict[str, Any]:
        """Run one mode and return its result dictionary ("error" key on failure)."""
        try:
            config = self._config(config_path, max_iterations, top_k, no_memory, db)
            if mode == "history":
                return self._history(config.memory.db_path, thread_id)
            if mode == "audit":
                return self._audit(config.memory.db_path, thread_id, full)
            loop = self._loop(config, thread_id)
            if mode == "chat":
                return self._chat(loop, new, output_format)
            result = loop.run_turn(prompt, new_conversation=new).to_dict()
            result["mode"] = "turn"
            return result
        except Exception as e:
            if self.debug:
                raise
            return {"mode": mode, "error": str(e), "prompt": prompt, "thread_id": thread_id}

    def _chat(self, loop, new: bool, output_format: str) -> Dict[str, Any]:
        stream = self.input_stream or sys.stdin
        out = self.output_stream or sys.stdout
        interactive = hasattr(stream, "isatty") and stream.isatty()
        turns = 0
        while True:
            if interactive:
                print("you> ", end="", file=out, flush=True)
            line = stream.readline()
            if not line:
                break
            question = line.strip()
            if not question:
                continue
            if question.lower() in EXIT_WORDS:
                break
            result = loop.run_turn(question, new_conversation=new and turns == 0).to_dict()
            result["mode"] = "turn"
            turns += 1
            print(self.format_output(result, output_format), file=out, flush=True)
        return {"mode": "chat", "thread_id": loop.thread_id, "turns": turns}

    def _history(self, db_path: str, thread_id: str) -> Dict[str, Any]:
        from sd_agentic_rag.agent.memory import AgentMemory
        turns = []
        if os.path.exists(db_path):
            with AgentMemory(db_path) as memory:
                turns = memory.history(thread_id)
        return {"mode": "history", "thread_id": thread_id, "db_path": db_path, "turns": turns}

    def _audit(self, db_path: str, thread_id: str, full: bool) -> Dict[str, Any]:
        from sd_agentic_rag.agent.memory import AgentMemory
        steps = []
        if os.path.exists(db_path):
            with AgentMemory(db_path) as memory:
                steps = memory.audit(thread_id)
        return {"mode": "audit", "thread_id": thread_id, "db_path": db_path, "full": full, "steps": steps}

    # -- output -------------------------------------------------------------

    def format_output(self, result: Dict[str, Any], output_format: str = "markdown") -> str:
        if output_format == "json":
            return json.dumps(result, indent=2, default=str)
        if "error" in result:
            return f"# ❌ Error\n\n```text\n{result['error']}\n```\n"
        mode = result.get("mode")
        if mode == "history":
            return self._format_history(result)
        if mode == "audit":
            return self._format_audit(result)
        if mode == "chat":
            return f"_Chat ended after {result.get('turns', 0)} turn(s) in thread `{result.get('thread_id')}`._"
        return self._format_turn(result)

    def _format_turn(self, result: Dict[str, Any]) -> str:
        lines = ["# Agent Answer", "", f"**Question:** {result.get('question', '')}", ""]
        lines += ["## Answer", "", result.get("answer", ""), ""]
        loop = result.get("loop", {})
        if result.get("partial"):
            lines += [f"> **Partial answer** - the budget ran out ({result.get('stop_reason')}); "
                      "it is grounded in the evidence found so far.", ""]
        citations = result.get("citations") or []
        if citations:
            lines += ["## Citations", ""]
            for c in citations:
                score = c.get("score")
                score_text = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                lines.append(f"- {c.get('ticket_id')} (score: {score_text}, retriever: {c.get('retriever')}, "
                             f"tool: {c.get('tool')})")
            lines.append("")
        guard = result.get("guard", {})
        lines += ["## Guard", "",
                  f"- Allowed: {guard.get('allowed')} | Enforced: {guard.get('enforced')} | "
                  f"Replaced: {guard.get('replaced')}"]
        for action in guard.get("actions") or []:
            lines.append(f"  - {action}")
        lines += ["", "## Loop", "",
                  f"- Iterations: {loop.get('iterations')} | LLM calls: {loop.get('llm_calls')} | "
                  f"Tokens: {loop.get('total_tokens')} | Elapsed: {loop.get('elapsed_s')}s | "
                  f"Stop reason: {loop.get('stop_reason')} | Partial: {result.get('partial')}"]
        if loop.get("fallbacks"):
            lines.append(f"- Fallbacks: {', '.join(loop['fallbacks'])}")
        if result.get("thread_id"):
            lines.append(f"- Thread: {result['thread_id']} (turn {result.get('turn_no')})")
        lines.append("")
        if self.verbose:
            lines += self._format_trace(result.get("trace") or [])
        return "\n".join(lines)

    @staticmethod
    def _describe(event: Dict[str, Any]) -> str:
        p = event.get("payload") or {}
        name = event.get("event")
        if name in ("plan", "refine"):
            goals = []
            open_ids = set(p.get("open_ids") or [])
            for g in p.get("sub_goals") or []:
                target = g.get("ticket_id") or g.get("query")
                extra = f" {g['filters']}" if g.get("filters") else ""
                mark = "" if g.get("id") in open_ids else " (done)"
                goals.append(f"{g.get('id')} {g.get('tool')} \"{target}\"{extra}{mark}")
            return f"{'Plan' if name == 'plan' else 'Refined plan'} ({p.get('planner')}): " + "; ".join(goals)
        if name == "tool_call":
            ids = ", ".join(p.get("evidence_ticket_ids") or []) or "no evidence"
            return f"Tool {p.get('sub_goal')} {p.get('tool')} -> {ids} ({p.get('count')} passages)"
        if name == "grounded":
            return f"Grounded: {', '.join(p.get('ticket_ids') or []) or 'nothing'}"
        if name == "verdict":
            gate = "Critic" if event.get("component") == "critic" else "Self-Review"
            ok = "sufficient" if gate == "Critic" else "passed"
            status = ok if p.get("passed") else ("insufficient" if gate == "Critic" else "failed")
            details = []
            if p.get("missing"):
                details.append(f"missing {', '.join(p['missing'])}")
            if p.get("refine_hint"):
                details.append(f"hint: {p['refine_hint']}")
            if p.get("reasons") and not p.get("passed"):
                details.append("; ".join(p["reasons"]))
            return f"{gate} ({p.get('mode')}): {status}" + (f" - {' | '.join(details)}" if details else "")
        if name == "draft":
            kind = "reserved best-effort draft" if p.get("reserved") else ("revision" if p.get("revision") else "draft")
            return f"Generate {kind} ({p.get('tokens')} tokens)"
        if name in ("planner_fallback", "critic_fallback", "self_review_llm_error", "planner_capped"):
            return f"Fallback {name}: {p.get('error', p)}"
        if name == "budget_exceeded":
            return f"Budget exceeded: {p.get('reason')}"
        if name == "exhaust":
            return f"Exhaustion ({p.get('stop_reason')}): best effort over {', '.join(p.get('grounded_ticket_ids') or []) or 'no evidence'}"
        if name == "context":
            return (f"Memory: {p.get('recent_turns')} recent turn(s), remembered "
                    f"{', '.join(p.get('remembered_ticket_ids') or []) or 'none'}, hints {', '.join(p.get('hints') or []) or 'none'}")
        if name == "final":
            return f"Guard: allowed={p.get('allowed')} replaced={p.get('replaced')} refused={p.get('refused')}"
        if name == "question":
            return "Guard: question PII redacted" if p.get("redacted") else "Guard: question checked"
        return f"{event.get('component')}.{name}: {json.dumps(p)[:120]}"

    def _format_trace(self, trace: List[Dict[str, Any]]) -> List[str]:
        lines = ["## Trace", ""]
        current = None
        for event in trace:
            iteration = event.get("iteration", 0)
            if iteration != current:
                current = iteration
                lines += ["", f"### {'Setup' if iteration == 0 else f'Iteration {iteration}'}", ""]
            lines.append(f"- {self._describe(event)}")
        lines.append("")
        return lines

    @staticmethod
    def _format_history(result: Dict[str, Any]) -> str:
        thread = result.get("thread_id")
        turns = result.get("turns") or []
        if not turns:
            return f"No conversation is saved for thread `{thread}`."
        lines = [f"# Conversation: thread `{thread}`", ""]
        for t in turns:
            flags = []
            if t.get("partial"):
                flags.append("partial")
            if t.get("refused"):
                flags.append("refused")
            lines += [f"## Turn {t['turn_no']}" + (f" ({', '.join(flags)})" if flags else ""), "",
                      f"**Q:** {t['question']}", "", f"**A:** {t['answer']}", ""]
            if t.get("cited_ticket_ids"):
                lines += [f"Cited: {', '.join(t['cited_ticket_ids'])}", ""]
        return "\n".join(lines)

    @staticmethod
    def _format_audit(result: Dict[str, Any]) -> str:
        thread = result.get("thread_id")
        steps = result.get("steps") or []
        if not steps:
            return f"No audit steps are saved for thread `{thread}`."
        full = result.get("full")
        lines = [f"# Audit: thread `{thread}`", ""]
        current = None
        for s in steps:
            if s["turn_no"] != current:
                current = s["turn_no"]
                epoch = s.get("epoch")
                lines += ["", f"## Turn {current}" + (f" (conversation {epoch})" if epoch is not None else ""), ""]
            payload = json.dumps(s.get("payload") or {}, default=str)
            if not full:
                payload = _short(payload, AUDIT_PREVIEW_CHARS)
            lines.append(f"- [{s['created_at']}] it{s['iteration']} #{s['seq']} "
                         f"{s['component']}.{s['event']}: {payload}")
        return "\n".join(lines)
