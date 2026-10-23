"""
Model Selection Module

This module provides automatic model selection based on the USE_OLLAMA_MODELS
environment variable. When Ollama mode is enabled, it uses the OpenAI-compatible
endpoint at OLLAMA_BASE_URL (default: http://localhost:11434/v1).

Configuration is read from two sources:
1. .env file (secrets only - API keys, passwords; model settings found
   there are reported as a warning because they would override the cfg)
2. sd-agentic-rag.cfg (application settings - model names, dimensions, URLs)

Both files support the following environment variables:

USE_OLLAMA_MODELS (bool, default: true)
  - true: Use Ollama models via OpenAI-compatible endpoint
  - false: Use OpenAI cloud models

OLLAMA_BASE_URL (str, default: http://localhost:11434/v1)
  - Base URL for Ollama's OpenAI-compatible API

OLLAMA_EMBEDDING_MODEL (str, default: nomic-embed-text)
  - Model name for embeddings when USE_OLLAMA_MODELS=true

OLLAMA_EMBEDDING_MODEL_DIMM (int, optional)
  - Dimension override for Ollama embedding vectors (used when USE_OLLAMA_MODELS=true);
    when unset, dimensions come from the known-model map or a one-time probe

OLLAMA_CHAT_MODEL (str, default: llama3)
  - Model name for LLM when USE_OLLAMA_MODELS=true

OLLAMA_API_KEY (optional)
  - API key for Ollama (if authentication is required)

OPENAI_API_KEY (required for OpenAI mode)
  - Your OpenAI API key

OPENAI_EMBEDDING_MODEL (str, default: text-embedding-3-small)
  - Model name for embeddings when USE_OLLAMA_MODELS=false

OPENAI_EMBEDDING_MODEL_DIMM (int, optional)
  - Dimension override for OpenAI embedding vectors (used when USE_OLLAMA_MODELS=false);
    when unset, dimensions come from the known-model map or a one-time probe

OPENAI_CHAT_MODEL (str, default: gpt-4o-mini)
  - Model name for LLM when USE_OLLAMA_MODELS=false

Command-line environment variables take precedence over both .env and sd-agentic-rag.cfg.
For example: USE_OLLAMA_MODELS=false uv run sd-agentic-rag ...
"""

import os
import logging
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv, dotenv_values

logger = logging.getLogger(__name__)

# config.yaml keys that would duplicate model identity owned by this module.
# They are ignored if present (see check_config_for_model_keys).
MODEL_CONFIG_KEYS = (
    ("embed", "embedding_model"),
    ("embed", "model"),
    ("openai", "embedding_model"),
    ("openai", "embedding_dimensions"),
    ("generate", "model"),
)

# Model settings owned by sd-agentic-rag.cfg. .env is loaded first, so one of
# these in .env would silently win over the cfg (see check_env_for_model_keys).
MODEL_ENV_KEYS = (
    "USE_OLLAMA_MODELS",
    "OLLAMA_BASE_URL",
    "OLLAMA_EMBEDDING_MODEL",
    "OLLAMA_EMBEDDING_MODEL_DIMM",
    "OLLAMA_CHAT_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_MODEL_DIMM",
    "OPENAI_CHAT_MODEL",
)


def check_env_for_model_keys(env_file: Path) -> List[str]:
    """Warn about model settings in .env, which should hold secrets only.

    Args:
        env_file: Path to the .env file

    Returns:
        List of offending keys (empty if none or the file is missing)
    """
    if not env_file.exists():
        return []
    values = dotenv_values(env_file)
    found = [key for key in MODEL_ENV_KEYS if key in values]
    if found:
        logger.warning(
            "%s contains model settings that override sd-agentic-rag.cfg: %s. "
            "Keep .env for secrets only and set these in sd-agentic-rag.cfg.",
            env_file, ", ".join(found)
        )
    return found


# Load secrets from .env (API keys, sensitive values)
# This file should NOT be committed to version control
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    check_env_for_model_keys(env_path)
    load_dotenv(env_path, override=False)

# Load configuration from sd-agentic-rag.cfg (model names, dimensions, URLs)
# This file contains application settings that may be version controlled
config_path = Path(__file__).parent.parent.parent / "sd-agentic-rag.cfg"
if config_path.exists():
    load_dotenv(config_path, override=False)

# Command-line environment variables still take precedence
# (they override both .env and sd-agentic-rag.cfg)


class ModelSelector:
    """Centralized model selection based on USE_OLLAMA_MODELS configuration."""

    def __init__(self):
        self._use_ollama = self._parse_bool(
            os.getenv("USE_OLLAMA_MODELS", "true")
        )

    @staticmethod
    def _parse_bool(value: str) -> bool:
        """Parse string to boolean, case-insensitive."""
        if isinstance(value, bool):
            return value
        return value.lower() in ("true", "1", "yes")

    @property
    def use_ollama(self) -> bool:
        """Return whether Ollama mode is enabled."""
        return self._use_ollama

    def get_embedding_model(self, default: Optional[str] = None) -> str:
        """
        Get the appropriate embedding model based on mode.

        Args:
            default: Fallback model name if none specified

        Returns:
            Model name string
        """
        if self._use_ollama:
            return os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def get_embedding_dimensions_override(self) -> Optional[int]:
        """
        Get the embedding dimension override for the active provider.

        Reads OLLAMA_EMBEDDING_MODEL_DIMM in Ollama mode, otherwise
        OPENAI_EMBEDDING_MODEL_DIMM. The other provider's key is ignored.

        Returns:
            Dimension count, or None if no override is set

        Raises:
            ValueError: If the override is set but is not a positive integer
        """
        var = "OLLAMA_EMBEDDING_MODEL_DIMM" if self._use_ollama else "OPENAI_EMBEDDING_MODEL_DIMM"
        raw = os.getenv(var)
        if raw is None or raw.strip() == "":
            return None
        try:
            dims = int(raw.strip())
        except ValueError:
            dims = 0
        if dims <= 0:
            raise ValueError(
                f"{var} must be a positive integer, got {raw!r}. "
                f"Fix it in .env or sd-agentic-rag.cfg, or unset it to use the model's default."
            )
        return dims

    def get_llm_model(self, default: Optional[str] = None) -> str:
        """
        Get the appropriate LLM model based on mode.

        Args:
            default: Fallback model name if none specified

        Returns:
            Model name string
        """
        if self._use_ollama:
            return os.getenv("OLLAMA_CHAT_MODEL", "llama3")
        return os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    def get_base_url(self) -> Optional[str]:
        """
        Get the base URL for API calls.

        Returns:
            Base URL string for Ollama-compatible API, or None for OpenAI
        """
        if self._use_ollama:
            return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return None

    def get_openai_api_key(self) -> str:
        """
        Get the API key (required for OpenAI mode, optional for Ollama).

        Returns:
            API key string

        Raises:
            ValueError: If in OpenAI mode and no API key is set
        """
        if not self._use_ollama:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required when "
                    "USE_OLLAMA_MODELS=false"
                )
            return api_key
        # Ollama API key is optional
        return os.getenv("OPENAI_API_KEY", "dummy-key-for-ollama")

    def get_client_kwargs(self) -> dict:
        """Get kwargs for initializing OpenAI client."""
        base_url = self.get_base_url()
        api_key = self.get_openai_api_key()

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        return kwargs


def create_model_selector() -> ModelSelector:
    """Factory function to create a ModelSelector instance."""
    return ModelSelector()


# Global singleton for easy access
_model_selector: Optional[ModelSelector] = None


def get_model_selector() -> ModelSelector:
    """Get the global ModelSelector instance."""
    global _model_selector
    if _model_selector is None:
        _model_selector = ModelSelector()
    return _model_selector


def use_ollama() -> bool:
    """Convenience function to check if Ollama mode is enabled."""
    return get_model_selector().use_ollama


def get_embedding_model_name() -> str:
    """Get the current embedding model name."""
    return get_model_selector().get_embedding_model()


def get_llm_model_name() -> str:
    """Get the current LLM model name."""
    return get_model_selector().get_llm_model()


def get_base_url() -> Optional[str]:
    """Get the current base URL for API calls."""
    return get_model_selector().get_base_url()


def get_client_kwargs() -> dict:
    """Get kwargs for initializing OpenAI client.

    Returns:
        Dictionary with api_key and optionally base_url for OpenAI client
    """
    selector = get_model_selector()
    base_url = selector.get_base_url()
    api_key = selector.get_openai_api_key()

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return kwargs


def check_config_for_model_keys(config: Optional[Dict], config_path: str = "config/config.yaml") -> List[str]:
    """Warn about model-identity keys in config.yaml, which are ignored.

    Model names and embedding dimensions are owned by this module
    (.env / sd-agentic-rag.cfg); config.yaml must not duplicate them.

    Args:
        config: Parsed config.yaml contents
        config_path: Path shown in the warning

    Returns:
        List of offending keys in dotted form (empty if none)
    """
    found = [
        f"{section}.{key}"
        for section, key in MODEL_CONFIG_KEYS
        if isinstance(config, dict)
        and isinstance(config.get(section), dict)
        and key in config[section]
    ]
    if found:
        logger.warning(
            "%s contains model settings that are ignored: %s. Model names and embedding "
            "dimensions are set in .env / sd-agentic-rag.cfg (USE_OLLAMA_MODELS, "
            "*_EMBEDDING_MODEL, *_CHAT_MODEL, *_EMBEDDING_MODEL_DIMM).",
            config_path, ", ".join(found)
        )
    return found


def print_model_config() -> None:
    """Print the current model configuration to stdout.

    Displays the active mode, embedding model, LLM model, and API endpoint
    being used by the RAG system.
    """
    selector = get_model_selector()

    mode = "OLLAMA (local)" if selector.use_ollama else "OPENAI (cloud)"
    base_url = selector.get_base_url()
    embedding_model = selector.get_embedding_model()
    llm_model = selector.get_llm_model()

    print("=" * 60)
    print("SdAgentic RAG - Model Configuration")
    print("=" * 60)
    print(f"Mode:              {mode}")
    print(f"Embedding Model:   {embedding_model}")
    print(f"LLM Model:         {llm_model}")
    if base_url:
        print(f"API Base URL:      {base_url}")
    else:
        print(f"API Base URL:      https://api.openai.com/v1 (default)")
    print("=" * 60)
