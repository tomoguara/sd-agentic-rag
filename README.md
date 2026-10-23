# SD Agentic RAG

A Production-Ready RAG (Retrieval-Augmented Generation) system for support desk tickets. Built with LangChain, OpenAI, and Qdrant, it helps answer customer support queries with grounded, cited answers from your ticket history.

---

**✨ Features**
- **7-Stage RAG Pipeline**: Ingest → Embed → Store → Retrieve → Re-rank → Ground → Generate
- **Hybrid Search**: Combines vector similarity with BM25 keyword matching
- **Citation-Verified Answers**: Every claim is traced back to source tickets
- **Smart Triage**: Automatically categorize and prioritize incoming tickets
- **Bulk Analysis**: Process and group multiple tickets
- **Guardrails**: Citation enforcement, relevance checking, PII redaction, and configurable refusal handling
- **Observability**: Full tracing and metrics collection
- **Evaluation Framework**: Measure answer quality, hallucination rates, and relevance

---

## Table of Contents

- [Part 1: User Guide - How to Use the System](#part-1-user-guide-how-to-use-the-system)
  - [Before You Start](#11-before-you-start)
  - [CLI Commands](#12-cli-commands)
  - [Examples](#13-examples)
- [Part 2: Developer Guide](#part-2-developer-guide)
  - [Setting Up Your Environment](#21-setting-up-your-environment)
  - [Architecture Overview](#22-architecture-overview)
  - [Module Reference](#23-module-reference)
  - [Testing](#24-testing)
  - [Model Configuration](#25-model-configuration)

---

# Part 1 - User Guide: How to Use the System

## 1.1 Before You Start

You need:

- **Python 3.14+** (use `pyenv` to install)
- `uv` for package management
- The project set up as described in [Setting up your environment](#21-setting-up-your-environment)
- Required API keys in a file named `.env` in the project folder

### Environment Setup

```bash
# Copy environment file and edit with your keys
cp .env.example .env
nano .env  # Add your OpenAI or Ollama configuration
```

Required environment variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...

# Or use Ollama (local models)
# USE_OLLAMA_MODELS=true
# OLLAMA_BASE_URL=http://localhost:11434/v1
# OLLAMA_EMBEDDING_MODEL=qwen3-embedding:latest
```

### Required Services

The system uses **Qdrant** as the vector database for storing and retrieving embeddings. Make sure Qdrant is running before using the system.

#### Starting Qdrant

```bash
# Start Qdrant container (in another terminal)
./scripts/qdrant-db.sh --start

# Wait for Qdrant to be ready (optional, auto-waits up to 30 seconds)
./scripts/qdrant-db.sh --wait

# Check status
./scripts/qdrant-db.sh --status

# Stop Qdrant when done
./scripts/qdrant-db.sh --stop
```

#### Configuration

The system uses a YAML configuration file at `config/config.yaml` with settings for:

- **OpenAI**: API key environment variable, embedding model, and batch size
- **Ingest**: Chunking parameters (size, overlap, strategy), PII cleaning, and chunk header options
- **Embed**: Embedding model and batch size
- **Store**: Qdrant connection settings (host, port, collection name prefix)
- **Retrieve**: Vector search weights and hybrid retrieval parameters
- **Re-rank**: Re-ranking model configuration
- **Ground**: Maximum passages and token limits
- **Guard**: Citation enforcement, relevance checking, PII redaction, and refusal handling
- **Logging**: Log level and format

##### Guard Configuration

The **Guard** section controls answer enforcement:

```yaml
guard:
  enable_citation_check: true      # Require citations in answers
  enable_relevance_gate: true      # Check relevance to query
  enable_pii_redact: true          # Redact personally identifiable information
  enable_safety_filter: false      # Detect unsafe/sensitive content
  enforce: true                    # Block non-compliant answers (use false for permissive mode)
  min_citations_per_answer: 1      # Minimum number of citations required
  max_tokens: 2000                 # Maximum answer length in tokens
  min_tokens: 10                   # Minimum answer length in tokens
  relevance_threshold: 0.5         # Minimum relevance score to pass
```

**Enforcement Behavior**:
- When `enforce: true`: Non-compliant answers are replaced with a standard refusal message
- When `enforce: false`: Answers pass through regardless of guard results
- **Refusals** ("I cannot find sufficient information...") are exempt from citation count and length checks
- **Output** includes a `guard` object with `allowed`, `enforced`, `replaced`, and `actions` fields

##### Per-Model Collections

The vector collection name is automatically suffixed with the embedding model name to support multiple embedding providers:

- **Default prefix**: `tickets`
- **Effective name format**: `tickets__<model-slug>`
- **Model slug**: Lowercased with non-alphanumeric characters replaced by `-`

**Examples**:
- `text-embedding-3-small` → `tickets__text-embedding-3-small` (1536 dimensions)
- `qwen3-embedding:latest` → `tickets__qwen3-embedding-latest` (4096 dimensions)
- Explicit `collection_name=` argument bypasses suffixing (useful for tests)

When a collection exists but has different dimensions than expected, the system raises a clear error with the collection name, actual dimensions, and the remedy (`ingest --wipe`).

##### Dimension Resolution

Embedding dimensions are determined automatically from the active model:

1. **Explicit override**: `openai.embedding_dimensions` in config (legacy, not recommended)
2. **Known models map**: Look up model name from `model_selector.get_embedding_model()`
3. **One-time probe**: Embed a short string and measure vector length

Known model dimensions:
- `text-embedding-3-small`: 1536
- `text-embedding-3-large`: 3072
- `text-embedding-ada-002`: 1536
- `nomic-embed-text`: 768
- `qwen3-embedding:latest`: 4096

**Note**: The explicit dimension override is deprecated. Use per-model collections instead.

#### Database Setup and Data Ingestion

Before you can answer questions, you need to load your ticket data into the vector database:

```bash
# Ingest tickets from the synthetic tickets file
sd-agentic-rag ingest --data-path data/synthetic_tickets.json

# For custom tickets, use your own JSON file with the same structure
sd-agentic-rag ingest --data-path data/your-tickets.json

# Wipe and re-ingest data (clears existing data first)
sd-agentic-rag ingest --data-path data/synthetic_tickets.json --wipe
```

**Data Format**: Your tickets file should be a JSON file with a `tickets` array containing ticket objects. Each ticket should have fields like `ticket_id`, `title`, `description`, `service`, `severity`, and `status`. See `data/synthetic_tickets.json` for the complete schema.

**Note**: If you see `Connection refused` errors, ensure Qdrant is running first with `./scripts/qdrant-db.sh --start`.

---

## 1.2 CLI Commands

The `sd-rag` CLI provides various commands for working with support tickets:

### Basic Usage

```bash
# Show all available commands
sd-agentic-rag --help

# Show help for a specific command
sd-agentic-rag answer --help
```

### Available Commands

| Command | Description |
|---------|-------------|
| `answer` | Answer support questions with citations |
| `evaluate` | Evaluate RAG responses with metrics against ground truth |
| `triage` | Automatically categorize and prioritize incoming tickets |
| `similar` | Find similar past tickets to the current issue |
| `escalate` | Generate escalation handoff summaries |
| `runbook` | Find runbook procedures for incidents |
| `gap` | Detect knowledge gaps in support tickets |
| `bulk` | Process and analyze multiple tickets at once |
| `ingest` | Load and index ticket data into the vector store |
| `qdb` | Query Qdrant database directly (debugging) |

### Command Details

#### `answer` - Answer Support Questions

```bash
# Basic usage
sd-agentic-rag answer --prompt "How do I reset my password?"

# With more results
sd-agentic-rag answer --prompt "Network timeout issues" --top-k 10

# JSON output for automation
sd-agentic-rag answer --prompt "Database connection failed" --json

# Verbose mode
sd-agentic-rag answer -p "Server downtime" -v
```

**Output**:
Markdown with citations like `[CITATION:INC001]`. Use `--json` for structured output:

```json
{
  "answer": {
    "answer": "...the actual answer text...",
    "citations": ["CITATION:INC001", "CITATION:INC002"]
  },
  "grounded_context": {
    "passages": [...],
    "citations": [
      {"ticket_id": "INC001", "score": 0.95, "retriever": "vector"}
    ]
  },
  "guard": {
    "allowed": true,
    "enforced": true,
    "replaced": false,
    "actions": []
  }
}
```

**Guard Fields**:
- `allowed`: Passes all guard checks
- `enforced`: Enforcement was active (`guard.enforce: true`), whether or not it fired
- `replaced`: The answer was blocked and replaced with the standard refusal message
- `actions`: List of guard checks applied

#### `evaluate` - Evaluate RAG Responses

```bash
# Basic evaluation
sd-agentic-rag evaluate --prompt "What is Python?"

# With ground truth for comparison
sd-agentic-rag evaluate \
  --prompt "What is Python?" \
  --ground-truth "Python is a programming language created by Guido van Rossum"
```

**Output**: Relevance score, answer quality, hallucination detection, context precision/recall, latency metrics.

#### `triage` - Ticket Triage and Categorization

```bash
# Triage a new ticket
sd-agentic-rag triage --prompt "Customer reports latency issues with API"

# Get JSON output for integration
sd-agentic-rag triage -p "Memory leak in production service" --json
```

**Output**:
- Summary of the issue
- Category: password/billing/technical/account/feature-request/other
- Urgency: high/medium/low
- Troubleshooting steps for the support agent

#### `similar` - Find Similar Cases

```bash
# Find similar cases to current issue
sd-agentic-rag similar --prompt "Database connection timeout"

# Find cases similar to a specific ticket
sd-agentic-rag similar --prompt "API rate limiting" --ticket-id INC-001

# Get more results
sd-agentic-rag similar --prompt "Authentication failure" --top-k 20
```

#### `escalate` - Generate Escalation Summaries

```bash
# Create escalation handoff
sd-agentic-rag escalate \
  --prompt "Server is down and cannot be restarted" \
  --context "Already tried basic troubleshooting, reboot, and logs check"

# With more context
sd-agentic-rag escalate \
  --prompt "Payment processing failed" \
  --context "User reported error after card update, tried 3 times"
```

**Output**: Issue summary, what's been tried, relevant logs, next team, action items.

#### `runbook` - Runbook Assistance

```bash
# Get runbook for incident
sd-agentic-rag runbook --prompt "Memory leak in production microservice"

# Find procedure for database issue
sd-agentic-rag runbook -p "Slow query performance on users table"
```

**Output**: Identify incident category, relevant runbook procedure with steps, critical warnings.

#### `gap` - Knowledge Gap Detection

```bash
# Analyze pattern for gaps
sd-agentic-rag gap --prompt "Frequent password reset issues"

# Analyze specific tickets
sd-agentic-rag gap \
  --prompt "Missing documentation for two-factor authentication" \
  --ticket-ids INC-001 INC-002 INC-003
```

**Output**: Missing information, documentation needs, top priorities.

#### `bulk` - Bulk Ticket Analysis

```bash
# Process multiple tickets
sd-agentic-rag bulk --file-path tickets.json

# Different analysis types
sd-agentic-rag bulk \
  --file-path tickets.json \
  --analysis-type group_by_symptom

# Group by product
sd-agentic-rag bulk -f tickets.json -a group_by_product
```

**Input format** (JSON):
```json
{
  "tickets": [
    {"prompt": "Issue 1 description"},
    {"prompt": "Issue 2 description"},
    {"description": "Issue 3 description"}
  ]
}
```

#### `ingest` - Index Ticket Data

```bash
# Ingest tickets from JSON file
sd-agentic-rag ingest --data-path data/tickets.json

# Don't clean PII (passwords, emails)
sd-agentic-rag ingest -d data/tickets.json --no-clean

# Wipe and re-ingest
sd-agentic-rag ingest -d data/tickets.json --wipe

# Disable chunk header (stores raw description only)
sd-agentic-rag ingest -d data/tickets.json --no-chunk-header
```

##### Fields Stored

When ingesting tickets, the following metadata fields are stored in the vector store:

- `ticket_id`: Unique ticket identifier
- `chunk_index`: Position of chunk within ticket
- `chunk_text`: Text content (with optional header prefix)
- `title`: Ticket title
- `service`: Service name
- `severity`: Severity level
- `ticket_type`: Type of ticket
- `status`: Current status
- `priority`: Priority level
- `sla_minutes`: SLA target in minutes
- `tags`: Array of tags
- `assigned_to`: Assignee name
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

**PII Policy**: User name and email are never stored in the vector store.

**Chunk Header**: By default, each chunk's text begins with a header containing ticket metadata (`Title: <title> | Service: <service> | Type: <ticket_type> | Tags: <t1, t2>`; empty fields are omitted). This helps with contextual retrieval while excluding volatile fields (status, priority, severity, SLA, assignee, dates). Disable with `--no-chunk-header` or `ingest.chunk_header: false` in `config.yaml` (the flag wins).

#### `qdb` - Query Qdrant Database

```bash
# Show collection information
sd-agentic-rag qdb --info

# List ticket IDs (default 20)
sd-agentic-rag qdb --list
sd-agentic-rag qdb --list 50

# Search by ticket ID
sd-agentic-rag qdb --ticket-id LAT001234

# Search by text similarity
sd-agentic-rag qdb --search "network timeout"

# Filter by metadata
sd-agentic-rag qdb --filter --service network --severity high
```

---

## 1.3 Examples

### Typical User Workflows

#### Workflow 1: Answering a Customer Question

```bash
# Step 1: Answer with citations
sd-agentic-rag answer \
  --prompt "How do I update my billing information?" \
  --top-k 5

# Step 2: Verify the answer quality
sd-agentic-rag evaluate \
  --prompt "How do I update my billing information?" \
  --ground-truth "Users can update billing info in Settings > Billing"
```

#### Workflow 2: Triage Incoming Tickets

```bash
# Process incoming tickets into a file
cat incoming_tickets.json | jq -c '.' | while read ticket; do
  sd-agentic-rag triage -p "$ticket" --json >> triaged.json
done
```

#### Workflow 3: Knowledge Gap Analysis

```bash
# Daily gap analysis
sd-agentic-rag gap \
  --prompt "Daily pattern analysis: what are users struggling with?"
```

#### Workflow 4: Bulk Processing End of Day

```bash
# Batch process all tickets
sd-agentic-rag bulk \
  --file-path /data/daily_tickets.json \
  --analysis-type group_by_product
```

### Common CLI Options

| Option | Description |
|--------|-------------|
| `--prompt` / `-p` | User's question or ticket description |
| `--json` | Output in JSON format (for automation) |
| `--verbose` / `-v` | Enable verbose output |
| `--debug` | Enable debug mode with stack trace |
| `--top-k N` | Number of results to retrieve (default: 5) |
| `--config PATH` | Override config file path |
| `--timeout SECS` | Request timeout in seconds |

---

# Part 2 - Developer Guide

## 2.1 Setting Up Your Environment

### Prerequisites

- **Python 3.14+** (use `pyenv` to install)
- `uv` for package management
- `docker` and `docker-compose` for local services (optional)

### Installation Steps

```bash
# 1. Clone the repository
cd /home/txo/codeio/my-public-ai-apps/sd-agentic-rag

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API keys
nano .env

# 3. Install dependencies
uv sync

# 4. Verify installation
uv run python -c "import sd_agentic_rag; print('Import successful')"
```

### Starting Local Services

#### Qdrant (Vector Database)

```bash
# Start with helper script
./scripts/qdrant-db.sh --start

# Or use Docker directly
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Check status
./scripts/qdrant-db.sh --status

# Stop when done
./scripts/qdrant-db.sh --stop
```

#### Ollama (Local Embeddings - Optional)

```bash
# Install Ollama if not already installed
# https://ollama.ai/download

# Pull required models
ollama pull qwen3-embedding:latest
ollama pull llama3

# Start Ollama
ollama serve
```

---

## 2.2 Architecture Overview

![SD Agentic RAG Full System Architecture](docs/images/full_system_architecture_readme.png)

*Click image to view full-size* (PlantUML source: [`full_system_architecture_readme.puml`](docs/images/full_system_architecture_readme.puml))

### Data Flow

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Ingest | Load tickets from JSON/CSV, clean PII, chunk text |
| 2 | Embed | Convert chunks to vectors using OpenAI or Ollama |
| 3 | Store | Persist vectors in Qdrant with metadata |
| 4 | Retrieve | Hybrid search (vector + BM25) with RRF fusion |
| 5 | Re-rank | Cross-encoder for precise relevance scoring |
| 6 | Ground | Assemble context with citation markers |
| 7 | Generate | LLM produces grounded answer |
| 8 | Guard | Verify citations, check relevance, redact PII |

---

## 2.3 Module Reference

### Core Pipeline Modules

| Module | Description | Main Classes |
|--------|-------------|--------------|
| `sd_agentic_ingest.py` | Ticket ingestion, chunking, deduplication | `SD AgenticIngest` |
| `sd_agentic_embed.py` | Embedding generation (OpenAI/Ollama) | `EmbeddingService`, `EmbeddingPipeline` |
| `sd_agentic_store.py` | Vector storage in Qdrant | `VectorStore`, `StorePipeline` |
| `sd_agentic_retrieve.py` | Hybrid search (vector + BM25) | `HybridRetriever`, `BM25Index` |
| `sd_agentic_rerank.py` | Cross-encoder re-ranking | `ReRanker`, `ReRankerPipeline` |
| `sd_agentic_ground.py` | Context assembly with citations | `ContextGrounding`, `GroundedContext` |
| `sd_agentic_generate.py` | LLM answer generation | `AnswerGenerator`, `GeneratedAnswer` |

### Guard and Observability Modules

| Module | Description | Main Classes |
|--------|-------------|--------------|
| `sd_agentic_guard.py` | Citation verification, PII redaction | `Guardrails` |
| `sd_agentic_observability.py` | Tracing and metrics collection | `RAGPipelineObserver`, `MetricsCollector` |
| `sd_agentic_eval.py` | Evaluation metrics and framework | `Evaluator`, `EvaluationPipeline` |

### CLI Command Modules

| Command | File | Purpose |
|---------|------|---------|
| `answer` | `rag_commands/rag_answer.py` | Full RAG pipeline for questions |
| `triage` | `rag_commands/rag_triage.py` | Ticket categorization |
| `similar` | `rag_commands/rag_similar.py` | Find similar cases |
| `escalate` | `rag_commands/rag_escalate.py` | Escalation handoffs |
| `runbook` | `rag_commands/rag_runbook.py` | Runbook assistance |
| `gap` | `rag_commands/rag_gap.py` | Knowledge gap detection |
| `bulk` | `rag_commands/rag_bulk.py` | Batch processing |
| `ingest` | `rag_commands/rag_ingest.py` | Data ingestion |
| `qdb` | `rag_commands/rag_qdb.py` | Qdrant debugging |

### Main Pipeline Class

**File**: `src/sd_agentic_rag/sd_agentic_rag_pipeline.py`

The `SD AgenticRAGPipeline` class integrates all stages:

```python
from sd_agentic_rag import SD AgenticRAGPipeline

# Initialize pipeline
pipeline = SD AgenticRAGPipeline()

# Run full RAG pipeline
result = pipeline.generate_answer(
    query_text="How do I reset my password?",
    top_k=5,
    rerank_enabled=True,
    ground_enabled=True
)

# Access results
print(result['answer'].answer)  # Answer text
print(result['answer'].citations)  # Source citations
print(result['grounded_context'])  # Used context
```

---

## 2.4 Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Unit tests only (no external dependencies)
uv run pytest tests/unit/

# Integration tests with real services
uv run pytest tests/integration/

# Functional tests (with Qdrant)
uv run pytest tests/functional/

# Specific test file
uv run pytest tests/test_sd_agentic_ingest.py -v

# Verbose output
uv run pytest tests/ -v -s

# Stop at first failure
uv run pytest -x
```

### Test Structure

```
tests/
├── unit/                    # Unit tests (mocked dependencies)
│   ├── test_ingest.py
│   ├── test_embed.py
│   └── ...
├── integration/             # Integration tests (real APIs)
│   ├── test_ingest.py
│   └── ...
├── functional/              # End-to-end tests
│   ├── test_store_real.py
│   ├── test_search_real.py
│   └── test_pipeline_real.py
└── test_data/               # Test fixtures and data
    └── synthetic_tickets.json
```

### Running Integration Tests

Some tests require Qdrant and Ollama running:

```bash
# Start services (in separate terminals)
./scripts/qdrant-db.sh --start
ollama serve

# Run functional tests
uv run pytest tests/functional/ -v

# Run specific test
uv run pytest tests/functional/test_pipeline_real.py -v -s
```

---

## 2.5 Model Configuration

### Environment Variables

The system supports multiple model providers based on `USE_OLLAMA_MODELS`:

| Variable | Ollama Mode | OpenAI Mode | Default |
|----------|-------------|-------------|---------|
| `USE_OLLAMA_MODELS` | `true` | `false` | `true` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | N/A | - |
| `OLLAMA_EMBEDDING_MODEL` | `qwen3-embedding:latest` | N/A | - |
| `OLLAMA_CHAT_MODEL` | `llama3` | N/A | - |
| `OPENAI_API_KEY` | N/A | Required | - |
| `OPENAI_EMBEDDING_MODEL` | N/A | `text-embedding-3-small` | - |
| `OPENAI_CHAT_MODEL` | N/A | `gpt-4o-mini` | - |

### Default Configuration

| Component | Ollama Model | OpenAI Model |
|-----------|-------------|--------------|
| **Embeddings** | `qwen3-embedding:latest` | `text-embedding-3-small` |
| **LLM** | `llama3` | `gpt-4o-mini` |
| **Dimensions** | 4096 | 1536 |

### Using OpenAI in Production

```bash
# Edit .env
cat > .env << EOF
USE_OLLAMA_MODELS=false
OPENAI_API_KEY=sk-your-key-here
# Optional: specify models
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# OPENAI_CHAT_MODEL=gpt-4o-mini
EOF

# Test
uv run python -c "from sd_agentic_rag import print_model_config; print_model_config()"
```

---

## 2.5 Migration Plan

### Per-Model Collections

When you switch embedding providers (e.g., from Ollama to OpenAI), the system creates new collections with model-specific names:

- **Ollama (qwen3-embedding)**: `tickets--qwen3-embedding:latest`
- **OpenAI (text-embedding-3-small)**: `tickets--text-embedding-3-small`
- **OpenAI (text-embedding-3-large)**: `tickets--text-embedding-3-large`

This isolation ensures no dimension conflicts occur between collections.

#### Migration Steps

1. **Check Current Collections**

```bash
# List all collections and their dimensions
sd-agentic-rag qdb --info
```

2. **Ingest Data with New Model**

```bash
# Wipe and re-ingest with the new model
cd /path/to/your/project
uv run sd-agentic-rag ingest \
  --data-path data/merged_tickets.json \
  --wipe
```

3. **Verify New Collection**

```bash
# Confirm the new collection exists with correct dimensions
sd-agentic-rag qdb --info
```

4. **Optional: Delete Old Collection**

```bash
# After verifying the new collection works, delete the old one
# (Only do this if you're sure the old model is no longer needed)

# First, get collection details
sd-agentic-rag qdb --info

# Then delete using Qdrant console or Python:
uv run python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
# Replace with your old collection name
old_collection = 'tickets--old-model-name'
client.delete_collection(old_collection)
print(f'Deleted collection: {old_collection}')
"
```

#### Dimension Override

If you need to override the default dimensions (advanced):

```bash
# In config.yaml, add this section:
embedding:
  dimensions: 2048  # Your custom dimension

# Or use environment variables:
export EMBEDDING_DIMENSIONS=2048
```

**Note**: Dimension override should only be used when your embedding provider returns a custom dimension that differs from the standard model output.

---

## 2.6 Common Extensions

### Adding a New CLI Command

1. Create `src/sd_agentic_rag/rag_commands/rag_<command>.py`
2. Implement `Rag<Command>Command` class with `run()` and `format_output()`
3. Register in `sd-rag` main function

### Adding a New Retriever

1. Implement in `src/sd_agentic_rag/sd_agentic_retrieve.py`
2. Register in `__init__.py`
3. Update `SD AgenticRAGPipeline` to use your retriever

---

## End-to-End Verification

This section documents the end-to-end testing performed for both embedding providers.

### Task 7.1: Dual-Provider Ingestion Verification

Both collections are created with correct vector dimensions:

| Provider | Collection Name | Dimensions |
|----------|-----------------|------------|
| Ollama (qwen3-embedding) | `tickets__qwen3-embedding-latest` | 4096 |
| OpenAI (text-embedding-3-small) | `tickets__text-embedding-3-small` | 1536 |

**Command verification**:

```bash
# Ollama mode (4096 dimensions)
USE_OLLAMA_MODELS=true uv run sd-agentic-rag ingest \
  --data-path data/merged_tickets.json --wipe --json

# OpenAI mode (1536 dimensions)  
USE_OLLAMA_MODELS=false uv run sd-agentic-rag ingest \
  --data-path data/merged_tickets.json --wipe --json
```

**Verification**:
- `sd-agentic-rag qdb --info` shows the correct collection name
- `sd-agentic-rag qdb --ticket-id LAT001234` displays `status`, `sla_minutes`, and no PII fields (user_name/user_email)

### Task 7.2: Dual-Provider Answer Generation Verification

Both providers return properly formatted answers with ticket IDs:

```bash
# Ollama mode
USE_OLLAMA_MODELS=true uv run sd-agentic-rag answer \
  --prompt "How was the high response latency on API endpoints fixed?" --json

# OpenAI mode
USE_OLLAMA_MODELS=false uv run sd-agentic-rag answer \
  --prompt "How was the high response latency on API endpoints fixed?" --json
```

**Verification**:
- `citations` array uses ticket IDs (e.g., `[CITATION:LAT001234]`), not point UUIDs
- `guard` block present with `allowed`, `enforced`, `replaced`, `actions` fields
- Answer is grounded in actual ticket content from the corpus

### Task 7.3: Functional Test Suite

Run the functional test suite for each provider:

```bash
# Both providers (OpenAI run skipped if key unset)
./scripts/run_functional.sh
```

Expected outputs:
- Each provider runs in a separate subprocess
- If `OPENAI_API_KEY` is unset, the OpenAI run is skipped with a clear message
- Both provider runs pass (81 tests each, using `pytest -m live`)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*This project is developed and maintained by Ton Oguara.*
