"""
Unit tests for Model Selection Module
"""
import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


from sd_agentic_rag.model_selector import (
    print_model_config,
    ModelSelector,
    create_model_selector,
    get_model_selector,
    use_ollama,
    get_embedding_model_name,
    get_llm_model_name,
    get_base_url,
    _model_selector
)


class TestModelSelector:
    """Tests for ModelSelector class"""

    def test_default_mode_is_ollama(self):
        """Test that default mode is Ollama (USE_OLLAMA_MODELS not set)"""
        # Clear USE_OLLAMA_MODELS
        if "USE_OLLAMA_MODELS" in os.environ:
            del os.environ["USE_OLLAMA_MODELS"]

        selector = ModelSelector()
        assert selector.use_ollama is True

    def test_ollama_mode_enabled(self):
        """Test that Ollama mode can be enabled"""
        selector = ModelSelector()
        # Simulate setting USE_OLLAMA_MODELS=true
        selector._use_ollama = True
        assert selector.use_ollama is True

    def test_parse_bool_variations(self):
        """Test boolean parsing with various inputs"""
        selector = ModelSelector()

        # Test various true values
        assert selector._parse_bool("true") is True
        assert selector._parse_bool("True") is True
        assert selector._parse_bool("TRUE") is True
        assert selector._parse_bool("1") is True
        assert selector._parse_bool("yes") is True
        assert selector._parse_bool(True) is True

        # Test various false values
        assert selector._parse_bool("false") is False
        assert selector._parse_bool("False") is False
        assert selector._parse_bool("FALSE") is False
        assert selector._parse_bool("0") is False
        assert selector._parse_bool("no") is False
        assert selector._parse_bool(False) is False

    def test_get_embedding_model_ollama(self):
        """Test embedding model in Ollama mode"""
        selector = ModelSelector()
        selector._use_ollama = True
        os.environ["OLLAMA_EMBEDDING_MODEL"] = "nomic-embed-text"

        model = selector.get_embedding_model()
        assert model == "nomic-embed-text"

    def test_get_embedding_model_openai(self):
        """Test embedding model in OpenAI mode"""
        selector = ModelSelector()
        selector._use_ollama = False
        os.environ["OPENAI_EMBEDDING_MODEL"] = "text-embedding-3-small"

        model = selector.get_embedding_model()
        assert model == "text-embedding-3-small"

    def test_get_embedding_model_defaults(self):
        """Test default embedding models"""
        selector = ModelSelector()

        # Ollama default
        selector._use_ollama = True
        if "OLLAMA_EMBEDDING_MODEL" in os.environ:
            del os.environ["OLLAMA_EMBEDDING_MODEL"]
        model = selector.get_embedding_model()
        assert model == "nomic-embed-text"

        # OpenAI default
        selector._use_ollama = False
        if "OPENAI_EMBEDDING_MODEL" in os.environ:
            del os.environ["OPENAI_EMBEDDING_MODEL"]
        model = selector.get_embedding_model()
        assert model == "text-embedding-3-small"

    def test_get_llm_model_ollama(self):
        """Test LLM model in Ollama mode"""
        selector = ModelSelector()
        selector._use_ollama = True
        os.environ["OLLAMA_CHAT_MODEL"] = "mistral"

        model = selector.get_llm_model()
        assert model == "mistral"

    def test_get_llm_model_openai(self):
        """Test LLM model in OpenAI mode"""
        selector = ModelSelector()
        selector._use_ollama = False
        os.environ["OPENAI_CHAT_MODEL"] = "gpt-4"

        model = selector.get_llm_model()
        assert model == "gpt-4"

    def test_get_llm_model_defaults(self):
        """Test default LLM models"""
        selector = ModelSelector()

        # Ollama default
        selector._use_ollama = True
        if "OLLAMA_CHAT_MODEL" in os.environ:
            del os.environ["OLLAMA_CHAT_MODEL"]
        model = selector.get_llm_model()
        assert model == "llama3"

        # OpenAI default
        selector._use_ollama = False
        if "OPENAI_CHAT_MODEL" in os.environ:
            del os.environ["OPENAI_CHAT_MODEL"]
        model = selector.get_llm_model()
        assert model == "gpt-4o-mini"

    def test_get_base_url_ollama(self):
        """Test base URL in Ollama mode"""
        selector = ModelSelector()
        selector._use_ollama = True
        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"

        url = selector.get_base_url()
        assert url == "http://localhost:11434/v1"

    def test_get_base_url_openai(self):
        """Test base URL in OpenAI mode (should be None)"""
        selector = ModelSelector()
        selector._use_ollama = False

        url = selector.get_base_url()
        assert url is None

    def test_get_base_url_default(self):
        """Test default Ollama base URL"""
        selector = ModelSelector()
        selector._use_ollama = True
        if "OLLAMA_BASE_URL" in os.environ:
            del os.environ["OLLAMA_BASE_URL"]

        url = selector.get_base_url()
        assert url == "http://localhost:11434/v1"

    def test_get_openai_api_key_required(self):
        """Test that API key is required in OpenAI mode"""
        selector = ModelSelector()
        selector._use_ollama = False

        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                selector.get_openai_api_key()
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_get_openai_api_key_optional_ollama(self):
        """Test that API key is optional in Ollama mode"""
        selector = ModelSelector()
        selector._use_ollama = True
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        api_key = selector.get_openai_api_key()
        assert api_key == "dummy-key-for-ollama"

    def test_get_client_kwargs_ollama(self):
        """Test client kwargs in Ollama mode"""
        selector = ModelSelector()
        selector._use_ollama = True
        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"

        kwargs = selector.get_client_kwargs()
        assert "api_key" in kwargs
        assert kwargs["base_url"] == "http://localhost:11434/v1"

    def test_get_client_kwargs_openai(self):
        """Test client kwargs in OpenAI mode"""
        selector = ModelSelector()
        selector._use_ollama = False
        os.environ["OPENAI_API_KEY"] = "test-key"

        kwargs = selector.get_client_kwargs()
        assert "api_key" in kwargs
        assert kwargs["api_key"] == "test-key"
        assert "base_url" not in kwargs


class TestGlobalFunctions:
    """Tests for global convenience functions"""

    def test_create_model_selector(self):
        """Test factory function"""
        selector = create_model_selector()
        assert isinstance(selector, ModelSelector)

    def test_get_model_selector_singleton(self):
        """Test global singleton behavior"""
        selector1 = get_model_selector()
        selector2 = get_model_selector()
        assert selector1 is selector2

    def test_use_ollama(self):
        """Test use_ollama returns the value from .env file"""
        # Reset the global selector to re-read .env
        if "_model_selector" in dir(sys.modules["sd_agentic_rag.model_selector"]):
            import importlib
            import sd_agentic_rag.model_selector as ms
            ms._model_selector = None
            importlib.reload(ms)

        # use_ollama() should reflect the value set in .env file
        import os
        use_ollama_val = use_ollama()

        # The value depends on what's in .env (either true or false)
        # This test just verifies the function works and returns a boolean
        assert isinstance(use_ollama_val, bool)

    def test_get_embedding_model_name(self):
        """Test get_embedding_model_name function"""
        model = get_embedding_model_name()
        assert isinstance(model, str)
        assert len(model) > 0

    def test_get_llm_model_name(self):
        """Test get_llm_model_name function"""
        model = get_llm_model_name()
        assert isinstance(model, str)
        assert len(model) > 0

    def test_get_base_url_function(self):
        """Test get_base_url convenience function"""
        url = get_base_url()
        # Either None (OpenAI mode) or a string (Ollama mode)
        assert url is None or isinstance(url, str)


class TestEnvironmentVariablePrecedence:
    """Test that .env file is properly loaded"""

    def test_env_file_loading(self):
        """Test that .env file variables are accessible"""
        import os

        # Check if .env variable exists
        from sd_agentic_rag.model_selector import load_dotenv

        # The load_dotenv should have been called at import time
        # so our .env variables should be accessible
        assert os.getenv("USE_OLLAMA_MODELS") is not None



class TestModelSelectorResetFixture:
    """Tests for the model_selector reset fixture"""

    def test_reset_switches_mode_to_ollama(self, reset_model_selector):
        """Test that reset can switch to Ollama mode"""
        reset_model_selector(mode='ollama')
        selector = ModelSelector()
        assert selector.use_ollama is True

    def test_reset_switches_mode_to_openai(self, reset_model_selector):
        """Test that reset can switch to OpenAI mode"""
        reset_model_selector(mode='openai')
        selector = ModelSelector()
        assert selector.use_ollama is False

    def test_reset_clears_env_and_uses_default(self, reset_model_selector):
        """Test that reset clears environment and uses default"""
        # Set to openai, then reset without mode
        os.environ['USE_OLLAMA_MODELS'] = 'false'
        reset_model_selector()
        selector = ModelSelector()
        # Default should be ollama when env var is not set
        assert selector.use_ollama is True


class TestPrintModelConfig:
    """Tests for print_model_config function"""

    def test_print_model_config_prints(self):
        """Test that print_model_config prints configuration"""
        import io
        import sys

        # Capture stdout
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            print_model_config()
            output = captured_output.getvalue()

            # Verify output contains expected elements
            assert "SdAgentic RAG - Model Configuration" in output
            assert "Mode:" in output
            assert "Embedding Model:" in output
            assert "LLM Model:" in output
        finally:
            sys.stdout = old_stdout


class TestConfigModelKeyCheck:
    """config.yaml must not duplicate model identity owned by model_selector"""

    def test_warns_and_lists_model_keys(self, caplog):
        from sd_agentic_rag.model_selector import check_config_for_model_keys
        config = {
            "embed": {"embedding_model": "text-embedding-3-small", "batch_size": 100},
            "openai": {"embedding_dimensions": 4096, "api_key_env": "OPENAI_API_KEY"},
            "generate": {"model": "gpt-4o-mini", "temperature": 0.2},
            "rerank": {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        }
        with caplog.at_level("WARNING", logger="sd_agentic_rag.model_selector"):
            found = check_config_for_model_keys(config, "config/config.yaml")
        assert found == ["embed.embedding_model", "openai.embedding_dimensions", "generate.model"]
        assert "ignored" in caplog.text and "embed.embedding_model" in caplog.text
        # The re-rank cross-encoder is pipeline behavior, not provider identity
        assert "rerank.model" not in caplog.text

    def test_clean_config_does_not_warn(self, caplog):
        from sd_agentic_rag.model_selector import check_config_for_model_keys
        with caplog.at_level("WARNING", logger="sd_agentic_rag.model_selector"):
            assert check_config_for_model_keys({"embed": {"batch_size": 100}}) == []
            assert check_config_for_model_keys(None) == []
        assert caplog.text == ""

    def test_shipped_config_has_no_model_keys(self):
        import yaml
        from sd_agentic_rag.model_selector import check_config_for_model_keys
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        with open(config_path) as f:
            assert check_config_for_model_keys(yaml.safe_load(f)) == []


def test_env_model_keys_are_reported(tmp_path, caplog):
    """Model settings in .env are reported; secrets are not"""
    import logging
    from sd_agentic_rag.model_selector import check_env_for_model_keys

    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-test\nOLLAMA_EMBEDDING_MODEL=nomic-embed-text\n")

    with caplog.at_level(logging.WARNING, logger="sd_agentic_rag.model_selector"):
        found = check_env_for_model_keys(env_file)

    assert found == ["OLLAMA_EMBEDDING_MODEL"]
    assert "OLLAMA_EMBEDDING_MODEL" in caplog.text
    assert "sk-test" not in caplog.text


def test_env_secrets_only_is_quiet(tmp_path, caplog):
    """A secrets-only .env (and a missing one) produces no warning"""
    import logging
    from sd_agentic_rag.model_selector import check_env_for_model_keys

    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-test\n")

    with caplog.at_level(logging.WARNING, logger="sd_agentic_rag.model_selector"):
        assert check_env_for_model_keys(env_file) == []
        assert check_env_for_model_keys(tmp_path / "missing.env") == []
    assert caplog.text == ""


def test_env_example_is_secrets_only():
    """.env.example ships no model settings that would override sd-agentic-rag.cfg"""
    from dotenv import dotenv_values
    from sd_agentic_rag.model_selector import MODEL_ENV_KEYS

    example = Path(__file__).parent.parent / ".env.example"
    assert not set(dotenv_values(example)) & set(MODEL_ENV_KEYS)
