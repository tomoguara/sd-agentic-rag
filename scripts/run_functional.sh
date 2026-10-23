#!/bin/bash
# run_functional.sh - Run functional tests for each provider
#
# This script runs pytest -m live tests/functional/ twice:
# 1. With USE_OLLAMA_MODELS=true (Ollama mode)
# 2. With USE_OLLAMA_MODELS=false (OpenAI mode)
#
# The OpenAI run is skipped with a clear message if OPENAI_API_KEY is unset.
#
# Usage: ./scripts/run_functional.sh

set -e

echo "=============================================="
echo "Functional Testing per Provider"
echo "=============================================="
echo ""

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading environment from .env..."
    # Source the .env file (simple export for key=value pairs, ignoring comments and empty lines)
    while IFS='=' read -r key value || [ -n "$key" ]; do
        # Skip empty lines and comments
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
        # Remove leading/trailing whitespace from key
        key=$(echo "$key" | xargs)
        # Remove surrounding quotes from value if present
        value=$(echo "$value" | sed 's/^["'\''"]//;s/["'\''"]$//')
        export "$key=$value"
    done < .env
fi

# Function to run tests for a specific provider
run_tests() {
    local provider=$1
    local use_ollama=$2
    local exit_code=0

    echo "----------------------------------------------"
    echo "Running tests for: $provider"
    echo "USE_OLLAMA_MODELS=$use_ollama"
    echo "----------------------------------------------"

    # If OpenAI mode and no API key, skip
    if [ "$provider" = "OpenAI" ]; then
        if [ -z "$OPENAI_API_KEY" ]; then
            echo ""
            echo "WARNING: OPENAI_API_KEY is not set, skipping OpenAI functional tests."
            echo "Set OPENAI_API_KEY to run OpenAI functional tests."
            echo ""
            return 0
        fi
    fi

    # Run tests with the specified provider
    USE_OLLAMA_MODELS=$use_ollama uv run pytest -m live tests/functional/ -v --tb=short || exit_code=$?

    echo ""
    echo "Summary for $provider: pytest exit code $exit_code"
    echo ""

    return $exit_code
}

# Run both providers even if the first fails (a bare call would trip `set -e`)
ollama_exit=0
run_tests "Ollama" "true" || ollama_exit=$?

openai_exit=0
run_tests "OpenAI" "false" || openai_exit=$?

echo "=============================================="
echo "Final Summary"
echo "=============================================="
echo "Ollama mode tests: $ollama_exit"
echo "OpenAI mode tests: $openai_exit"
echo ""

# Exit with error if both failed and OpenAI was attempted
if [ $ollama_exit -ne 0 ] && [ $openai_exit -ne 0 ]; then
    echo "ERROR: Both provider test runs failed."
    exit 1
elif [ $ollama_exit -ne 0 ]; then
    echo "NOTE: Ollama tests failed but OpenAI tests passed (if attempted)."
    exit $ollama_exit
elif [ $openai_exit -ne 0 ]; then
    echo "NOTE: OpenAI tests failed but Ollama tests passed."
    exit $openai_exit
else
    echo "SUCCESS: All provider test runs passed."
    exit 0
fi