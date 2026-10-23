# Real Integration Tests for Stage 3: Storage

This directory contains real integration tests that connect to a live Qdrant instance.

## Prerequisites

These tests require a running Qdrant instance at `localhost:6333`.

### Starting Qdrant with Docker

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or run Qdrant locally:
```bash
# Download Qdrant from https://github.com/qdrant/qdrant
# Run:
qdrant
```

## Test Files

- `test_store_real.py` - Tests for storing and managing vectors
- `test_search_real.py` - Tests for semantic search and filtering
- `test_pipeline_real.py` - End-to-end pipeline tests
- `test_generate_real.py` - End-to-end answer generation tests
- `test_guard_real.py` - Guardrail enforcement tests
- `test_ground_real.py` - Context grounding tests
- `test_eval_real.py` - Evaluation metric tests
- `test_rerank_real.py` - Re-ranker tests
- `test_retrieve_real.py` - Retrieval tests

## Running Tests

Functional tests carry the `live` marker, and the default `pytest` run
deselects them (`addopts = "-m 'not live'"` in `pyproject.toml`), so plain
`uv run pytest` is the offline unit suite. **Pass `-m live` to run them**;
without it, `pytest tests/functional/` collects nothing.

### With Qdrant Available
```bash
uv run pytest -m live tests/functional/ -v --tb=short
```

### Show why any test was skipped
```bash
uv run pytest -m live tests/functional/ -v -rs
```

### Skip the Qdrant connection check (if not available)
```bash
uv run pytest -m live tests/functional/ -v --tb=short --deselect tests/functional/test_store_real.py::TestQdrantConnection
```

### Provider-Specific Functional Tests

Functional tests are marked with `@pytest.mark.live` and require provider credentials. Run them using the provided script:

```bash
./scripts/run_functional.sh
```

This script:
- Runs tests for Ollama mode (`USE_OLLAMA_MODELS=true`)
- Runs tests for OpenAI mode (`USE_OLLAMA_MODELS=false`)
- Skips OpenAI tests with a clear message if `OPENAI_API_KEY` is unset
- Reports per-provider pass/fail status

**Environment variables:**
- `USE_OLLAMA_MODELS=true` - Use Ollama (local)
- `USE_OLLAMA_MODELS=false` - Use OpenAI (cloud)
- `OPENAI_API_KEY` - Required for OpenAI mode
- `OLLAMA_BASE_URL` - Ollama endpoint (default: `http://localhost:11434/v1`)

## Test Categories

### Store Tests (`test_store_real.py`)
- `TestVectorStoreReal` - Basic storage operations
  - `test_create_collection_and_store_single_chunk` - Store single vector
  - `test_store_multiple_chunks` - Batch storage
  - `test_batch_storage_with_different_batch_sizes` - Different batch sizes
  - `test_dimension_validation` - Validate embedding dimensions
  - `test_store_empty_list` - Handle empty input

- `TestStorePipelineReal` - StorePipeline operations
  - `test_store_and_verify_payload` - Verify stored data
  - `test_collection_exists_check` - Collection management
  - `test_store_pipeline_returns_correct_format` - API contract

### Search Tests (`test_search_real.py`)
- `TestSemanticSearchReal` - Basic search functionality
  - `test_semantic_search_with_related_query` - Similar query search
  - `test_search_with_metadata_filter` - Filtered search
  - `test_search_returns_top_k_results` - Top-k precision
  - `test_search_with_zero_results` - Edge cases

- `TestSearchWithRealEmbeddings` - Real embedding workflow
  - `test_search_with_real_embeddings_workflow` - Full workflow

### Pipeline Tests (`test_pipeline_real.py`)
- `TestFullPipelineIntegration` - End-to-end pipeline
  - `test_full_pipeline_ingest_embed_store` - Full pipeline run
  - `test_search_after_pipeline_storage` - Search after storage
  - `test_pipeline_with_multiple_severities` - Complex filtering

## Test Notes

- Tests use unique collection names (with UUID + timestamp) to avoid conflicts
- Collections are deleted after each test, except in `test_ground_real.py`, whose
  teardown calls a `VectorStore.delete_collection()` that does not exist and so
  leaves `test_ground_*` collections behind (tracked in
  `docs/harden-base-pipeline-review.md`)
- A test skips only when a service it needs (Qdrant, Ollama, OpenAI) is
  unreachable - see `service_checks.py`. Any other setup error (import error,
  bad argument, dimension mismatch) fails the test instead of skipping it, so
  with all services up a run should report **zero skips** (check with `-rs`)
