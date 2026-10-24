#!/usr/bin/env python3
"""SdAgentic RAG CLI - Main entry point.

This is the main CLI entry point for the SdAgentic RAG system.
It provides sub-commands for various RAG operations.

Usage:
    sd-agentic-rag answer --prompt "Your question here"
    sd-agentic-rag agent --prompt "Is LAT001234 resolved and what was its SLA?"
    sd-agentic-rag evaluate --prompt "Your question here" --ground-truth "Expected answer"
    sd-agentic-rag triage --prompt "Ticket description"
    sd-agentic-rag similar --prompt "Current issue" --ticket-id REF-123
    sd-agentic-rag escalate --prompt "Issue summary" --context "What's been tried"
    sd-agentic-rag runbook --prompt "Incident description"
    sd-agentic-rag gap --prompt "Pattern analysis"
    sd-agentic-rag bulk --file-path tickets.json
    sd-agentic-rag ingest --data-path data.json
    sd-agentic-rag qdb --list
    sd-agentic-rag qdb --ticket-id TICKET-123

Advanced options:
    --config PATH    Override config file path
    --json           Output in JSON format
    --verbose/-v     Enable verbose output
    --debug          Enable debug mode (implies verbose)
    --timeout SECS   Request timeout in seconds
    --top-k N        Number of results to retrieve (default: 5)
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# For development, allow running directly from src directory
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent))
    __package__ = "sd_agentic_rag"

from .rag_commands import (
    RagAnswerCommand,
    RagTriageCommand,
    RagSimilarCommand,
    RagIngestCommand,
    RagEvaluateCommand,
    RagBulkCommand,
    RagEscalateCommand,
    RagRunbookCommand,
    RagGapCommand,
)
from .sd_agentic_rag_pipeline import SdAgenticRAGPipeline


AGENT_EPILOG = """
Examples:
  sd-agentic-rag agent --prompt "Is LAT001234 resolved and what was its SLA?"
  sd-agentic-rag agent --prompt "Why are API calls slow?" --verbose
  sd-agentic-rag agent --thread-id demo --prompt "What was its resolution?"
  sd-agentic-rag agent --thread-id demo --new --prompt "Replica lag on DB01?"
  sd-agentic-rag agent --chat --thread-id demo
  sd-agentic-rag agent --history --thread-id demo
  sd-agentic-rag agent --audit --thread-id demo --full
  sd-agentic-rag agent --prompt "Why are API calls slow?" --max-iterations 1 --no-memory --json
"""


def _positive_int(value: str) -> int:
    """argparse type: an integer >= 1."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer >= 1, got {value!r}")
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be an integer >= 1, got {value!r}")
    return number


def validate_agent_args(parser: argparse.ArgumentParser, args) -> None:
    """Usage errors for option combinations the mode group cannot express."""
    if args.full and not args.audit:
        parser.error("--full requires --audit")
    if args.no_memory and (args.history or args.audit):
        parser.error("--no-memory cannot be combined with --history or --audit")
    if args.chat and args.json:
        parser.error("--json is not supported with --chat")
    if (args.history or args.audit) and (args.new or args.max_iterations is not None or args.top_k is not None):
        parser.error("--new, --max-iterations and --top-k apply to --prompt and --chat only")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must not be empty")


def validate_evaluate_args(parser: argparse.ArgumentParser, args) -> None:
    """Usage errors for evaluate option combinations."""
    if args.golden_file is None and (args.compare or args.export):
        parser.error("--compare and --export require --golden-file")
    if args.golden_file is not None and (args.agentic or args.ground_truth):
        parser.error("--agentic and --ground-truth apply to --prompt only (a golden run uses the agent)")


def run_agent_command(args) -> int:
    """Run `agent` and return the exit code (0 for answers, partial answers and refusals)."""
    from .rag_commands.rag_agent import RagAgentCommand

    mode = "turn" if args.prompt is not None else "chat" if args.chat else "history" if args.history else "audit"
    output_format = "json" if args.json else "markdown"
    command = RagAgentCommand(verbose=args.verbose or args.debug, debug=args.debug)
    try:
        result = command.run(
            mode=mode,
            prompt=args.prompt,
            thread_id=args.thread_id,
            new=args.new,
            max_iterations=args.max_iterations,
            top_k=args.top_k,
            no_memory=args.no_memory,
            db=args.db,
            config_path=args.config,
            full=args.full,
            output_format=output_format,
        )
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    print(command.format_output(result, output_format))
    return 1 if "error" in result else 0


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="sd-agentic-rag",
        description="SdAgentic RAG CLI - Answer questions, triage tickets, and more",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sd-agentic-rag answer --prompt "How do I reset my password?"
  sd-agentic-rag agent --prompt "Is LAT001234 resolved and what was its SLA?"
  sd-agentic-rag agent --chat --thread-id demo
  sd-agentic-rag evaluate --prompt "What is Python?" --ground-truth "Python is a language"
  sd-agentic-rag triage --prompt "Customer reports latency issues"
  sd-agentic-rag similar --prompt "Database connection timeout" --ticket-id INC-001
  sd-agentic-rag escalate --prompt "Server down" --context "Already restarted once"
  sd-agentic-rag runbook --prompt "Memory leak in production"
  sd-agentic-rag gap --prompt "Frequent password reset issues"
  sd-agentic-rag bulk --file-path /path/to/tickets.json
  sd-agentic-rag ingest --data-path /path/to/data.json
  sd-agentic-rag qdb --list
  sd-agentic-rag qdb --list 50
  sd-agentic-rag qdb --ticket-id LAT001234
  
Advanced options:
  --config PATH    Override config file path
  --json           Output in JSON format
  --verbose/-v     Enable verbose output
  --debug          Enable debug mode (implies verbose)
  --timeout SECS   Request timeout in seconds
  --top-k N        Number of results to retrieve
        """
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND"
    )
    
    # answer command
    answer_parser = subparsers.add_parser(
        "answer",
        help="Answer support questions with citations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-agentic-rag answer --prompt 'How do I reset my password?'"
    )
    answer_parser.add_argument("--prompt", "-p", required=True, help="User's question")
    answer_parser.add_argument("--config", type=str, default=None, help="Config file path")
    answer_parser.add_argument("--top-k", type=int, default=5, help="Results to retrieve")
    answer_parser.add_argument("--json", action="store_true", help="Output as JSON")
    answer_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    answer_parser.add_argument("--debug", action="store_true", help="Debug mode")
    answer_parser.add_argument("--timeout", type=int, default=None, help="Request timeout")
    
    # agent command
    agent_parser = subparsers.add_parser(
        "agent",
        help="Answer through the agentic loop (plan, retrieve, check, revise)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=AGENT_EPILOG
    )
    agent_mode = agent_parser.add_mutually_exclusive_group(required=True)
    agent_mode.add_argument("--prompt", "-p", help="Answer one question (single turn)")
    agent_mode.add_argument("--chat", action="store_true", help="Interactive chat: one turn per line until exit/quit")
    agent_mode.add_argument("--history", action="store_true", help="Show the thread's current conversation")
    agent_mode.add_argument("--audit", action="store_true", help="Show every stored loop step of the thread")
    agent_parser.add_argument("--full", action="store_true", help="With --audit: show untruncated step text")
    agent_parser.add_argument("--thread-id", default="default", help="Conversation thread (default: default)")
    agent_parser.add_argument("--new", action="store_true", help="Start a fresh conversation in the thread")
    agent_parser.add_argument("--max-iterations", type=_positive_int, default=None,
                              help="Loop iterations (>= 1; 1 = base pipeline; default: agent.budget.max_iterations)")
    agent_parser.add_argument("--top-k", type=_positive_int, default=None, help="Passages per tool call (default: agent.top_k)")
    agent_parser.add_argument("--no-memory", action="store_true", help="Neither read nor write agent memory")
    agent_parser.add_argument("--db", default=None,
                              help="Memory database (default: $SD_AGENTIC_MEMORY_DB or agent.memory.db_path)")
    agent_parser.add_argument("--config", type=str, default=None, help="Config file path")
    agent_parser.add_argument("--json", action="store_true", help="Output as JSON (single turn, history, audit)")
    agent_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output (per-iteration trace)")
    agent_parser.add_argument("--debug", action="store_true", help="Debug mode")

    # evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate RAG responses with metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  sd-agentic-rag evaluate --prompt 'What is Python?' --ground-truth 'Python is a language'
  sd-agentic-rag evaluate --prompt 'Is LAT001234 resolved?' --agentic --json
  sd-agentic-rag evaluate --golden-file data/eval/agentic_golden.json --compare --export results.json"""
    )
    eval_source = eval_parser.add_mutually_exclusive_group(required=True)
    eval_source.add_argument("--prompt", "-p", help="Question to evaluate")
    eval_source.add_argument("--golden-file", type=str, default=None,
                             help="Golden query set to evaluate (runs the agent; see --compare)")
    eval_parser.add_argument("--ground-truth", "-g", type=str, default=None, help="Expected answer")
    eval_parser.add_argument("--compare", action="store_true",
                             help="With --golden-file: also run the single-pass answer baseline and show deltas")
    eval_parser.add_argument("--export", type=str, default=None, metavar="PATH",
                             help="With --golden-file: write per-query results and aggregates as JSON")
    eval_parser.add_argument("--agentic", action="store_true",
                             help="With --prompt: evaluate through the agent instead of answer")
    eval_parser.add_argument("--config", type=str, default=None, help="Config file path")
    eval_parser.add_argument("--top-k", type=int, default=5, help="Results to retrieve")
    eval_parser.add_argument("--json", action="store_true", help="Output as JSON")
    eval_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    eval_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # triage command
    triage_parser = subparsers.add_parser(
        "triage",
        help="Triage support tickets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-agentic-rag triage --prompt 'Customer reports latency issues'"
    )
    triage_parser.add_argument("--prompt", "-p", required=True, help="Ticket description")
    triage_parser.add_argument("--config", type=str, default=None, help="Config file path")
    triage_parser.add_argument("--top-k", type=int, default=5, help="Similar cases to retrieve")
    triage_parser.add_argument("--json", action="store_true", help="Output as JSON")
    triage_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    triage_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # similar command
    similar_parser = subparsers.add_parser(
        "similar",
        help="Find similar support cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-agentic-rag similar --prompt 'Database connection timeout' --ticket-id INC-001"
    )
    similar_parser.add_argument("--prompt", "-p", required=True, help="Current issue description")
    similar_parser.add_argument("--ticket-id", "-t", type=str, default=None, help="Ticket ID to compare")
    similar_parser.add_argument("--config", type=str, default=None, help="Config file path")
    similar_parser.add_argument("--top-k", type=int, default=10, help="Similar cases to retrieve")
    similar_parser.add_argument("--json", action="store_true", help="Output as JSON")
    similar_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    similar_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest data into the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-agentic-rag ingest --data-path data.json"
    )
    ingest_parser.add_argument("--data-path", "-d", required=True, help="Path to data file")
    ingest_parser.add_argument("--config", type=str, default=None, help="Config file path")
    ingest_parser.add_argument("--no-clean", action="store_false", dest="clean", default=True, help="Don't clean PII")
    ingest_parser.add_argument("--wipe", action="store_true", default=False, help="Wipe collection before ingesting")
    ingest_parser.add_argument("--no-wipe", action="store_false", dest="wipe", default=False, help="Keep existing data (default)")
    ingest_parser.add_argument("--no-chunk-header", action="store_false", dest="chunk_header", default=None, help="Don't prepend the ticket header to chunk text (overrides ingest.chunk_header)")
    ingest_parser.add_argument("--json", action="store_true", help="Output as JSON")
    ingest_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    ingest_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # escalate command
    escalate_parser = subparsers.add_parser(
        "escalate",
        help="Generate escalation handoff summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-agentic-rag escalate --prompt 'Server down' --context 'Already restarted'"
    )
    escalate_parser.add_argument("--prompt", "-p", required=True, help="Issue summary")
    escalate_parser.add_argument("--context", "-c", type=str, default=None, help="What's been tried")
    escalate_parser.add_argument("--config", type=str, default=None, help="Config file path")
    escalate_parser.add_argument("--top-k", type=int, default=5, help="Results to retrieve")
    escalate_parser.add_argument("--json", action="store_true", help="Output as JSON")
    escalate_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    escalate_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # runbook command
    runbook_parser = subparsers.add_parser(
        "runbook",
        help="Get runbook assistance for incidents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-agentic-rag runbook --prompt 'Memory leak in production'"
    )
    runbook_parser.add_argument("--prompt", "-p", required=True, help="Incident description")
    runbook_parser.add_argument("--config", type=str, default=None, help="Config file path")
    runbook_parser.add_argument("--top-k", type=int, default=5, help="Runbooks to retrieve")
    runbook_parser.add_argument("--json", action="store_true", help="Output as JSON")
    runbook_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    runbook_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # gap command
    gap_parser = subparsers.add_parser(
        "gap",
        help="Detect knowledge gaps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-agentic-rag gap --prompt 'Frequent password reset issues'"
    )
    gap_parser.add_argument("--prompt", "-p", required=True, help="Pattern analysis")
    gap_parser.add_argument("--ticket-ids", "-t", type=str, nargs='+', default=None, help="Ticket IDs to analyze")
    gap_parser.add_argument("--config", type=str, default=None, help="Config file path")
    gap_parser.add_argument("--top-k", type=int, default=5, help="Results to retrieve")
    gap_parser.add_argument("--json", action="store_true", help="Output as JSON")
    gap_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    gap_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # bulk command
    bulk_parser = subparsers.add_parser(
        "bulk",
        help="Bulk ticket analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-agentic-rag bulk --file-path tickets.json"
    )
    bulk_parser.add_argument("--file-path", "-f", required=True, help="Path to tickets JSON file")
    bulk_parser.add_argument("--analysis-type", "-a", type=str, default="group_by_product",
                            choices=["group_by_product", "group_by_symptom", "summarize"],
                            help="Type of analysis")
    bulk_parser.add_argument("--config", type=str, default=None, help="Config file path")
    bulk_parser.add_argument("--top-k", type=int, default=5, help="Results per ticket")
    bulk_parser.add_argument("--json", action="store_true", help="Output as JSON")
    bulk_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    bulk_parser.add_argument("--debug", action="store_true", help="Debug mode")

    # qdb command
    qdb_parser = subparsers.add_parser(
        "qdb",
        help="Query Qdrant database directly for debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sd-agentic-rag qdb --info
  sd-agentic-rag qdb --list
  sd-agentic-rag qdb --list 50
  sd-agentic-rag qdb --ticket-id LAT001234
  sd-agentic-rag qdb --search "latency issues"
  sd-agentic-rag qdb --filter --service "Database Services"
        """
    )
    qdb_parser.add_argument("--info", action="store_true", help="Show collection information")
    qdb_parser.add_argument("--list", "-l", nargs='?', const=20, type=int, help="List ticket IDs (default: 20, or specify number)")
    qdb_parser.add_argument("--ticket-id", "-t", type=str, help="Search for specific ticket ID")
    qdb_parser.add_argument("--search", "-s", type=str, help="Search by text similarity")
    qdb_parser.add_argument("--filter", "-f", action="store_true", help="Filter by metadata")
    qdb_parser.add_argument("--service", type=str, help="Filter by service (use with --filter)")
    qdb_parser.add_argument("--severity", type=str, help="Filter by severity (use with --filter)")
    qdb_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results for search")
    qdb_parser.add_argument("--json", action="store_true", help="Output in JSON")
    qdb_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    qdb_parser.add_argument("--debug", action="store_true", help="Debug mode")

    # Parse arguments
    args = parser.parse_args()
    
    # If no command provided, show help
    if args.command is None:
        # Check if --prompt was provided without subcommand
        if hasattr(args, 'prompt') and args.prompt:
            # Run full pipeline with prompt
            run_full_pipeline(args)
            return
        print(parser.format_help())
        sys.exit(0)
    
    if args.command == "agent":
        validate_agent_args(agent_parser, args)
        sys.exit(run_agent_command(args))
    if args.command == "evaluate":
        validate_evaluate_args(eval_parser, args)

    # Run the appropriate command
    command_map = {
        "answer": ("rag_answer", "RagAnswerCommand"),
        "evaluate": ("rag_evaluate", "RagEvaluateCommand"),
        "triage": ("rag_triage", "RagTriageCommand"),
        "similar": ("rag_similar", "RagSimilarCommand"),
        "ingest": ("rag_ingest", "RagIngestCommand"),
        "escalate": ("rag_escalate", "RagEscalateCommand"),
        "runbook": ("rag_runbook", "RagRunbookCommand"),
        "gap": ("rag_gap", "RagGapCommand"),
        "bulk": ("rag_bulk", "RagBulkCommand"),
        "qdb": ("rag_qdb", "RagQdbCommand"),
    }
    
    if args.command not in command_map:
        parser.print_help()
        sys.exit(1)
    
    module_name, class_name = command_map[args.command]
    
    try:
        # Import the command module
        command_module = __import__(
            f"sd_agentic_rag.rag_commands.{module_name}",
            fromlist=[class_name]
        )
        command_class = getattr(command_module, class_name)
        
        # Get command-specific arguments (map config to config_path, exclude json_output)
        command_args = {}
        for k, v in vars(args).items():
            if k in ['command', 'debug', 'verbose']:
                continue
            # Map 'config' to 'config_path' for command modules
            if k == 'config':
                command_args['config_path'] = v
            # Skip json/json_output as it's handled by format_output
            elif k in ['json', 'json_output']:
                continue
            # Skip timeout since run() doesn't accept it
            elif k == 'timeout':
                continue
            # Handle list parameter - if it's True (default), use 20, otherwise use the value
            elif k == 'list' and v is True:
                command_args['list'] = 20
            else:
                command_args[k] = v
        
        # Initialize and run command
        command = command_class(
            verbose=args.verbose or args.debug,
            debug=args.debug
        )
        
        # Run the command
        result = command.run(**command_args)
        
        # Format and print output
        output_format = "json" if args.json else "markdown"
        output = command.format_output(result, output_format)
        print(output)
        
        # Exit with appropriate code
        if "error" in result:
            sys.exit(1)
        sys.exit(0)
        
    except ImportError as e:
        print(f"[ERROR] Failed to import command module: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


def run_full_pipeline(args):
    """Run the full pipeline when --prompt is provided without subcommand."""
    from . import SdAgenticRAGPipeline
    
    print(f"[INFO] Running full RAG pipeline for: {args.prompt}")
    
    pipeline = SdAgenticRAGPipeline()
    
    try:
        result = pipeline.generate_answer(
            query_text=args.prompt,
            top_k=args.top_k if hasattr(args, 'top_k') else 5,
            ground_enabled=True
        )
        
        # Format output
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            # Markdown format
            print("# SD Agentic RAG Response")
            print("")
            print(f"**Question:** {args.prompt}")
            print("")
            
            answer_obj = result.get("answer")
            if answer_obj:
                if isinstance(answer_obj, dict):
                    answer = answer_obj.get("answer", "")
                else:
                    answer = getattr(answer_obj, "answer", "")
                print("## Answer")
                print("")
                print(answer)
                print("")
            
            # Guard verdict
            guard_result = result.get('guard_result', {})
            guard_allowed = guard_result.get('allowed', True)
            guard_enforced = guard_result.get('enforced', False)
            guard_replaced = guard_result.get('replaced', False)
            guard_actions = guard_result.get('actions', [])
            
            if args.json:
                # Add guard info to JSON output
                result['guard'] = {
                    'allowed': guard_allowed,
                    'enforced': guard_enforced,
                    'replaced': guard_replaced,
                    'actions': guard_actions
                }
            else:
                # Show guard info in markdown when replaced or in verbose mode
                if guard_replaced or (args.verbose and guard_result):
                    print("## Guard Status")
                    print("")
                    print(f"- Allowed: {guard_allowed}")
                    print(f"- Enforced: {guard_enforced}")
                    print(f"- Replaced: {guard_replaced}")
                    if guard_actions:
                        print("- Actions:")
                        for action in guard_actions:
                            print(f"  - {action}")
                    print("")
            
            # Citations
            grounded_context = result.get("grounded_context")
            if grounded_context:
                citations = getattr(grounded_context, "citations", [])
                if citations:
                    print("## Citations")
                    print("")
                    for citation in citations:
                        ticket_id = citation.get('ticket_id', 'N/A')
                        score = citation.get('score', 'N/A')
                        retriever = citation.get('retriever', 'N/A')
                        print(f"- [{ticket_id}] (Score: {score:.3f}, Retriever: {retriever})")
                    print("")
        
        sys.exit(0)
        
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
