"""The pipeline's Guard is built from config.yaml's guard: section."""

from unittest.mock import MagicMock, patch

import pytest

from sd_agentic_rag import sd_agentic_rag_pipeline as pipeline_module
from sd_agentic_rag.sd_agentic_guard import Config as GuardConfig


@pytest.fixture
def guard_config(tmp_path):
    """Point the Guard Config singleton at a temp config.yaml."""
    def _write(yaml_text):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_text)
        GuardConfig._instance = None
        GuardConfig(config_path=str(config_file))
    yield _write
    GuardConfig._instance = None


@pytest.fixture
def mocked_stages():
    """Replace every external stage so the pipeline can be built offline."""
    names = [
        "SdAgenticIngest", "EmbeddingService", "EmbeddingPipeline", "VectorStore",
        "StorePipeline", "create_hybrid_retriever", "ReRanker", "ReRankerPipeline",
        "ContextGrounding", "AnswerGenerator",
    ]
    patches = [patch.object(pipeline_module, name, MagicMock()) for name in names
               if hasattr(pipeline_module, name)]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


def build_pipeline(**kwargs):
    return pipeline_module.SdAgenticRAGPipeline(observability_enabled=False, **kwargs)


def test_pipeline_guard_uses_config(guard_config, mocked_stages):
    guard_config(
        "guard:\n"
        "  enforce: false\n"
        "  relevance_threshold: 0.8\n"
        "  min_citations_per_answer: 2\n"
        "  min_tokens: 5\n"
        "  max_tokens: 2000\n"
    )

    guard = build_pipeline().guard_pipeline

    assert guard.enforce is False
    assert guard.relevance_threshold == 0.8
    assert guard.min_citations == 2
    assert guard.min_tokens == 5
    # Answer length comes from guard.max_tokens, not the grounding budget (4000)
    assert guard.max_tokens == 2000


def test_pipeline_args_override_config(guard_config, mocked_stages):
    guard_config("guard:\n  enforce: false\n  relevance_threshold: 0.8\n")

    guard = build_pipeline(guard_enforce=True, relevance_threshold=0.3).guard_pipeline

    assert guard.enforce is True
    assert guard.relevance_threshold == 0.3


def test_enforce_off_returns_blocked_answer(guard_config, mocked_stages):
    """With guard.enforce: false a blocked answer passes through, verdict still reported."""
    guard_config(
        "guard:\n"
        "  enforce: false\n"
        "  enable_relevance_gate: false\n"
        "  min_citations_per_answer: 1\n"
    )
    guard = build_pipeline().guard_pipeline

    answer = "Restart the auth service and clear the session cache to fix it."
    result = guard.process_answer(answer, "How do I fix login?", [{"ticket_id": "INC001"}])

    assert result["allowed"] is False
    assert result["enforced"] is False
    assert result["modified_answer"] == answer
