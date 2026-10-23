from sd_agentic_rag.sd_agentic_ingest import SdAgenticIngest
from sd_agentic_rag.sd_agentic_embed import (
    Config,
    EmbeddingService,
    EmbeddingPipeline,
    create_embedding_pipeline,
    SdAgenticRAGPipeline as SdAgenticRAGPipelineEmbed,
)
from sd_agentic_rag.sd_agentic_rag_pipeline import SdAgenticRAGPipeline
from sd_agentic_rag.sd_agentic_store import (
    VectorStore,
    StorePipeline,
    create_vector_store,
    create_store_pipeline,
    get_embedding_model_dimensions,
)
from sd_agentic_rag.sd_agentic_retrieve import (
    Config as RetrieveConfig,
    BM25Index,
    HybridRetriever,
    create_hybrid_retriever,
    create_bm25_index,
    SdAgenticRetriever,
)
from sd_agentic_rag.sd_agentic_rerank import (
    Config as ReRankConfig,
    ReRanker,
    ReRankerPipeline,
    create_reranker,
    SdAgenticReRanker,
    SdAgenticReRankerPipeline,
)
from sd_agentic_rag.sd_agentic_ground import (
    GroundedContext,
    Config as GroundConfig,
    ContextGrounding,
    create_context_grounder,
    SdAgenticGrounding,
    SdAgenticGroundedContext,
)
from sd_agentic_rag.model_selector import (
    ModelSelector,
    create_model_selector,
    get_model_selector,
    use_ollama,
    get_embedding_model_name,
    get_llm_model_name,
    get_base_url,
    print_model_config
)

# Backward compatibility import
from sd_agentic_rag.sd_agentic_embed import SdAgenticRAGPipelineWithStore

# Stage 7 imports
from sd_agentic_rag.sd_agentic_generate import (
    GeneratedAnswer,
    Config as GenerateConfig,
    AnswerGenerator,
    create_answer_generator,
    SdAgenticGenerator,
    SdAgenticGeneratedAnswer,
)

# Guard imports
from sd_agentic_rag.sd_agentic_guard import (
    Guardrails,
    Config as GuardConfig,
    create_guardrails,
    SdAgenticGuardrails,
    SdAgenticGuard,
)

# Observability imports
from sd_agentic_rag.sd_agentic_observability import (
    Config as ObservabilityConfig,
    trace_stage,
    RAGPipelineObserver,
    MetricsCollector,
    ObservabilityPipeline,
    create_observability_pipeline,
    SdAgenticObservability,
    SdAgenticMetricsCollector,
    SdAgenticPipelineObserver,
)

# Evaluation imports
from sd_agentic_rag.sd_agentic_eval import (
    EvaluationResult,
    EvaluationSummary,
    Evaluator,
    EvaluationPipeline,
    EvaluationMetric,
    create_evaluator,
    create_evaluation_pipeline,
    compute_basic_metrics,
)

__all__ = [
    "SdAgenticIngest",
    "Config",
    "SdAgenticRAGPipeline",
    "SdAgenticRAGPipelineWithStore",
    "VectorStore",
    "StorePipeline",
    "create_vector_store",
    "create_store_pipeline",
    "get_embedding_model_dimensions",
    "BM25Index",
    "HybridRetriever",
    "create_hybrid_retriever",
    "create_bm25_index",
    "SdAgenticRetriever",
    "RetrieveConfig",
    "ReRanker",
    "ReRankerPipeline",
    "create_reranker",
    "SdAgenticReRanker",
    "SdAgenticReRankerPipeline",
    "ReRankConfig",
    "GroundedContext",
    "GroundConfig",
    "ContextGrounding",
    "create_context_grounder",
    "SdAgenticGrounding",
    "SdAgenticGroundedContext",
    "ModelSelector",
    "create_model_selector",
    "get_model_selector",
    "use_ollama",
    "get_embedding_model_name",
    "get_llm_model_name",
    "get_base_url",
    "print_model_config",
    # Stage 7 exports
    "GeneratedAnswer",
    "GenerateConfig",
    "AnswerGenerator",
    "create_answer_generator",
    "SdAgenticGenerator",
    "SdAgenticGeneratedAnswer",
    # Guard exports
    "Guardrails",
    "GuardConfig",
    "create_guardrails",
    "SdAgenticGuardrails",
    "SdAgenticGuard",
    # Observability exports
    "ObservabilityConfig",
    "trace_stage",
    "RAGPipelineObserver",
    "MetricsCollector",
    "ObservabilityPipeline",
    "create_observability_pipeline",
    "SdAgenticObservability",
    "SdAgenticMetricsCollector",
    "SdAgenticPipelineObserver",
    # Evaluation exports
    "EvaluationResult",
    "EvaluationSummary",
    "Evaluator",
    "EvaluationPipeline",
    "EvaluationMetric",
    "create_evaluator",
    "create_evaluation_pipeline",
    "compute_basic_metrics",
]


def main() -> None:
    print("Hello from sd-agentic-rag!")
