"""RAG Evaluate Command - Evaluate RAG responses.

This module implements the evaluate command for the SdAgentic RAG CLI.
It runs evaluation metrics on RAG responses with optional ground truth.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RagEvaluateCommand:
    """Command to evaluate RAG responses."""

    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose or debug
        self.debug = debug

    def run(self, prompt: Optional[str] = None, ground_truth: Optional[str] = None,
            config_path: Optional[str] = None, top_k: int = 5,
            golden_file: Optional[str] = None, compare: bool = False,
            export: Optional[str] = None, agentic: bool = False,
            pipeline: Any = None) -> Dict[str, Any]:
        """Run the evaluation command.
        
        Args:
            prompt: User's question (mutually exclusive with golden_file)
            ground_truth: Optional ground truth answer for comparison
            config_path: Optional path to config file
            top_k: Number of results to retrieve
            golden_file: Golden query set to run (agent by default)
            compare: With golden_file, also run the single-pass `answer` baseline
            export: With golden_file, write full results as JSON to this path
            agentic: With prompt, evaluate through the agent instead of `answer`
            pipeline: Optional pre-built SdAgenticRAGPipeline (tests)
            
        Returns:
            Dictionary with evaluation results
        """
        if (prompt is None) == (golden_file is None):
            return {"error": "Give exactly one of --prompt or --golden-file"}
        if golden_file is None and (compare or export):
            return {"error": "--compare and --export require --golden-file"}
        if golden_file is not None:
            return self.run_golden(golden_file, compare=compare, export=export,
                                   config_path=config_path, top_k=top_k, pipeline=pipeline)

        from sd_agentic_rag import (
            SdAgenticRAGPipeline,
            create_evaluator,
            EvaluationPipeline
        )
        
        if self.verbose:
            print("[INFO] Initializing RAG Pipeline and Evaluator...")
        
        # Initialize RAG pipeline
        rag_pipeline = pipeline or SdAgenticRAGPipeline()
        
        # Initialize evaluator
        evaluator = create_evaluator()

        if agentic:
            return self._run_agentic(rag_pipeline, evaluator, prompt, ground_truth, config_path, top_k)
        
        # Evaluate the query
        try:
            if self.verbose:
                print("[INFO] Running RAG pipeline and evaluation...")
            
            # Create evaluation pipeline
            eval_pipeline = EvaluationPipeline(
                rag_pipeline=rag_pipeline,
                evaluator=evaluator
            )
            
            # Evaluate single query
            eval_result = eval_pipeline.evaluate_query(
                query_id="eval-001",
                query_text=prompt,
                ground_truth=ground_truth
            )
            
            # Get evaluation result
            result = {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "predicted_answer": eval_result.predicted_answer,
                "relevance_score": eval_result.relevance_score,
                "answer_quality_score": eval_result.answer_quality_score,
                "hallucination_score": eval_result.hallucination_score,
                "context_precision": eval_result.context_precision,
                "context_recall": eval_result.context_recall,
                "latency_ms": eval_result.latency_ms,
                "model": eval_result.model,
                "timestamp": eval_result.timestamp
            }
            
            return result
            
        except Exception as e:
            if self.debug:
                raise
            return {
                "error": str(e),
                "prompt": prompt,
                "ground_truth": ground_truth,
                "predicted_answer": "",
                "relevance_score": 0.0,
                "answer_quality_score": 0.0,
                "hallucination_score": 1.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "latency_ms": 0.0
            }

    def _run_agentic(self, rag_pipeline, evaluator, prompt, ground_truth, config_path, top_k) -> Dict[str, Any]:
        """Evaluate one query through the agentic loop (memory disabled)."""
        import time
        from sd_agentic_rag.agent import AgentConfig, AgenticRAGLoop
        from sd_agentic_rag.model_selector import get_llm_model_name

        try:
            config = AgentConfig.load(config_path, top_k=top_k, no_memory=True)
            start = time.time()
            turn = AgenticRAGLoop(rag_pipeline, config).run_turn(prompt)
            latency_ms = (time.time() - start) * 1000
            eval_result = evaluator.evaluate(
                query_id="eval-001", query_text=prompt, predicted_answer=turn.answer,
                ground_truth=ground_truth, retrieved_context=turn.grounded, latency_ms=latency_ms
            )
            return {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "predicted_answer": eval_result.predicted_answer,
                "relevance_score": eval_result.relevance_score,
                "answer_quality_score": eval_result.answer_quality_score,
                "hallucination_score": eval_result.hallucination_score,
                "context_precision": eval_result.context_precision,
                "context_recall": eval_result.context_recall,
                "latency_ms": eval_result.latency_ms,
                "model": get_llm_model_name(),
                "timestamp": eval_result.timestamp,
                "agentic": True,
                "citations": turn.citations,
                "partial": turn.partial,
                "loop": turn.loop,
            }
        except Exception as e:
            if self.debug:
                raise
            return {"error": str(e), "prompt": prompt, "agentic": True}

    def run_golden(self, golden_file: str, compare: bool = False, export: Optional[str] = None,
                   config_path: Optional[str] = None, top_k: int = 5, pipeline: Any = None) -> Dict[str, Any]:
        """Run a golden query set through the agent (and the baseline with compare)."""
        from sd_agentic_rag.sd_agentic_eval import (
            GoldenSetEvaluator,
            load_golden_set,
            make_agent_runner,
            make_baseline_runner,
        )
        from sd_agentic_rag.agent import AgentConfig
        from sd_agentic_rag.model_selector import get_embedding_model_name, get_llm_model_name, use_ollama

        try:
            entries = load_golden_set(golden_file)
            if pipeline is None:
                import contextlib
                from sd_agentic_rag import SdAgenticRAGPipeline
                with contextlib.redirect_stdout(sys.stderr):
                    pipeline = SdAgenticRAGPipeline()
                if not pipeline.store_enabled:
                    raise RuntimeError("Vector store is not available - is Qdrant running?")
            runners = {}
            if compare:
                runners["baseline"] = make_baseline_runner(pipeline, top_k=top_k)
            runners["agent"] = make_agent_runner(pipeline, AgentConfig.load(config_path, top_k=top_k))
            progress = (lambda msg: print(f"[INFO] {msg}", file=sys.stderr)) if self.verbose else None
            result = GoldenSetEvaluator(runners).evaluate(entries, progress=progress)
        except Exception as e:
            if self.debug:
                raise
            return {"error": str(e), "golden_file": golden_file}

        result.update({
            "golden_file": golden_file,
            "compare": compare,
            "provider": "ollama" if use_ollama() else "openai",
            "chat_model": get_llm_model_name(),
            "embedding_model": get_embedding_model_name(),
        })
        if export:
            Path(export).parent.mkdir(parents=True, exist_ok=True)
            with open(export, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            result["exported_to"] = export
        return result

    @staticmethod
    def _fmt(value) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def format_golden(self, result: Dict[str, Any]) -> str:
        """Markdown tables: one row per query and runner, one aggregate row per tag."""
        fmt = self._fmt
        runners = list(result["aggregates"])
        lines = ["# Golden-Set Evaluation", "",
                 f"**Golden file:** {result['golden_file']}  ",
                 f"**Provider:** {result['provider']} | **Chat model:** {result['chat_model']} | "
                 f"**Embedding model:** {result['embedding_model']}", ""]
        lines += ["## Per query", "",
                  "| ID | Tag | Runner | Recall | Precision | Refusal OK | Iterations | LLM calls | Tokens | Latency (ms) | Partial | Cited |",
                  "|----|-----|--------|--------|-----------|------------|------------|-----------|--------|--------------|---------|-------|"]
        for row in result["per_query"]:
            for name in runners:
                r = row["results"][name]
                if "error" in r:
                    lines.append(f"| {row['id']} | {row['tag']} | {name} | error: {r['error'][:60]} |||||||||")
                    continue
                lines.append(
                    f"| {row['id']} | {row['tag']} | {name} | {fmt(r['recall'])} | {fmt(r['precision'])} | "
                    f"{fmt(r['refusal_correct'])} | {r['iterations']} | {r['llm_calls']} | {r['total_tokens']} | "
                    f"{r['latency_ms']:.0f} | {fmt(r['partial'])} | {', '.join(r['cited_ticket_ids']) or '-'} |")
        lines += ["", "## Per tag", ""]
        if "deltas" in result:
            lines += ["| Tag | Queries | Recall (base / agent / Δ) | Precision (base / agent / Δ) | "
                      "Refusal OK (base / agent / Δ) | Agent iterations | Agent LLM calls | Agent partial |",
                      "|-----|---------|------|------|------|------|------|------|"]
            base, agent = result["aggregates"]["baseline"], result["aggregates"]["agent"]
            for tag, delta in result["deltas"].items():
                b, a = base[tag], agent[tag]
                cells = [f"{fmt(b[m])} / {fmt(a[m])} / {fmt(delta[m])}" for m in ("recall", "precision", "refusal_correct")]
                lines.append(f"| {tag} | {a['queries']} | " + " | ".join(cells) +
                             f" | {fmt(a['iterations'])} | {fmt(a['llm_calls'])} | {fmt(a['partial_rate'])} |")
        else:
            lines += ["| Runner | Tag | Queries | Recall | Precision | Refusal OK | Iterations | LLM calls | Tokens | Latency (ms) | Partial rate | Fallbacks |",
                      "|--------|-----|---------|--------|-----------|------------|------------|-----------|--------|--------------|--------------|-----------|"]
            for name, by_tag in result["aggregates"].items():
                for tag, a in by_tag.items():
                    lines.append(f"| {name} | {tag} | {a['queries']} | {fmt(a['recall'])} | {fmt(a['precision'])} | "
                                 f"{fmt(a['refusal_correct'])} | {fmt(a['iterations'])} | {fmt(a['llm_calls'])} | "
                                 f"{fmt(a['total_tokens'])} | {fmt(a['latency_ms'])} | {fmt(a['partial_rate'])} | "
                                 f"{a['fallbacks']} |")
        if result.get("exported_to"):
            lines += ["", f"Full results exported to `{result['exported_to']}`."]
        lines.append("")
        return "\n".join(lines)

    def format_output(self, result: Dict[str, Any], output_format: str = "markdown") -> str:
        """Format the output for display."""
        if output_format == "json":
            return json.dumps(result, indent=2, default=str)
        if "per_query" in result and "error" not in result:
            return self.format_golden(result)
        
        # Markdown format
        lines = []
        lines.append("# RAG Evaluation Results")
        lines.append("")
        
        if "error" in result:
            lines.append("## ❌ Error")
            lines.append("")
            lines.append(f"```text\n{result['error']}\n```")
            return "\n".join(lines)
        
        lines.append(f"**Question:** {result.get('prompt', 'N/A')}")
        lines.append("")
        
        predicted = result.get('predicted_answer', '')
        if predicted:
            lines.append("## Predicted Answer")
            lines.append("")
            lines.append(predicted)
            lines.append("")
        
        ground_truth = result.get('ground_truth')
        if ground_truth:
            lines.append("## Ground Truth")
            lines.append("")
            lines.append(ground_truth)
            lines.append("")
        
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        
        metrics = [
            ("Relevance", result.get('relevance_score')),
            ("Answer Quality", result.get('answer_quality_score')),
            ("Hallucination", result.get('hallucination_score')),
            ("Context Precision", result.get('context_precision')),
            ("Context Recall", result.get('context_recall')),
            ("Latency (ms)", result.get('latency_ms')),
        ]
        
        for name, value in metrics:
            if value is not None:
                score_str = f"{value:.3f}" if isinstance(value, float) else str(value)
                lines.append(f"| {name} | {score_str} |")
        
        lines.append("")
        
        if result.get('model'):
            lines.append(f"**Model:** {result.get('model')}")
            lines.append("")

        loop = result.get('loop')
        if loop:
            lines.append("## Agent Loop")
            lines.append("")
            lines.append(f"- Iterations: {loop.get('iterations')} | LLM calls: {loop.get('llm_calls')} | "
                         f"Tokens: {loop.get('total_tokens')} | Stop reason: {loop.get('stop_reason')} | "
                         f"Partial: {result.get('partial')}")
            lines.append("")
        
        if result.get('timestamp'):
            lines.append(f"**Timestamp:** {result.get('timestamp')}")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point for the evaluate command."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG responses with metrics"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="User's question to evaluate"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        default=None,
        help="Ground truth answer for comparison"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output in JSON format"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (implies verbose)"
    )
    
    args = parser.parse_args()
    
    command = RagEvaluateCommand(
        verbose=args.verbose,
        debug=args.debug
    )
    
    result = command.run(
        prompt=args.prompt,
        ground_truth=args.ground_truth,
        config_path=args.config,
        top_k=args.top_k
    )
    
    output_format = "json" if args.json_output else "markdown"
    output = command.format_output(result, output_format)
    print(output)
    
    if "error" in result:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
