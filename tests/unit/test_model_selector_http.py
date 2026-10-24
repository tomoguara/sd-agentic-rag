"""HTTP timeout and retry settings for every OpenAI-compatible client (design D15)."""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import openai
import pytest

import sd_agentic_rag.model_selector as ms


@pytest.fixture
def selector(monkeypatch):
    monkeypatch.setenv("USE_OLLAMA_MODELS", "true")
    monkeypatch.delenv("HTTPX_TIMEOUT", raising=False)
    monkeypatch.delenv("HTTPX_MAX_RETRIES", raising=False)
    original = ms._model_selector
    ms._model_selector = None
    yield ms.get_model_selector()
    ms._model_selector = original


def test_defaults(selector):
    kwargs = ms.get_client_kwargs()
    assert kwargs["timeout"] == 120.0
    assert kwargs["max_retries"] == 2


def test_overrides(selector, monkeypatch):
    monkeypatch.setenv("HTTPX_TIMEOUT", "7.5")
    monkeypatch.setenv("HTTPX_MAX_RETRIES", "4")
    kwargs = ms.get_client_kwargs()
    assert kwargs["timeout"] == 7.5
    assert kwargs["max_retries"] == 4


@pytest.mark.parametrize("var, value", [
    ("HTTPX_TIMEOUT", "0"), ("HTTPX_TIMEOUT", "-3"), ("HTTPX_TIMEOUT", "soon"),
    ("HTTPX_MAX_RETRIES", "0"), ("HTTPX_MAX_RETRIES", "-1"), ("HTTPX_MAX_RETRIES", "2.5"),
])
def test_invalid_values_name_the_variable(selector, monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError, match=var):
        ms.get_client_kwargs()


class _SlowHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        time.sleep(3)
        try:
            self.send_response(200)
            self.end_headers()
        except Exception:
            pass

    def log_message(self, *args):
        pass


def test_request_exceeding_timeout_raises_api_timeout_error(selector, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("OLLAMA_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        monkeypatch.setenv("HTTPX_TIMEOUT", "0.3")
        monkeypatch.setenv("HTTPX_MAX_RETRIES", "1")
        client = openai.OpenAI(**ms.get_client_kwargs())
        start = time.monotonic()
        with pytest.raises(openai.APITimeoutError):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        # Two attempts of 0.3 s plus a short backoff - nowhere near the library's 600 s default
        assert time.monotonic() - start < 5
    finally:
        server.shutdown()


def test_every_client_is_built_from_get_client_kwargs(selector):
    sentinel = {"api_key": "k", "timeout": 1.0, "max_retries": 1}
    with patch("sd_agentic_rag.model_selector.get_client_kwargs", return_value=sentinel), \
            patch("openai.OpenAI") as client_cls:
        client_cls.return_value = MagicMock()

        from sd_agentic_rag.sd_agentic_embed import EmbeddingService
        with patch("sd_agentic_rag.sd_agentic_store.get_embedding_model_dimensions", return_value=3):
            EmbeddingService(model="m")
        client_cls.assert_called_with(**sentinel)
        client_cls.reset_mock()

        from sd_agentic_rag.sd_agentic_generate import AnswerGenerator
        AnswerGenerator(model="m")
        client_cls.assert_called_with(**sentinel)
        client_cls.reset_mock()

        from sd_agentic_rag.sd_agentic_store import _dimension_probe
        client_cls.return_value.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.0] * 5)])
        if hasattr(_dimension_probe, "_cache"):
            _dimension_probe._cache.pop("probe-model", None)
        assert _dimension_probe("probe-model") == 5
        client_cls.assert_called_with(**sentinel)
        client_cls.reset_mock()

        import sd_agentic_rag.agent.llm as llm
        original = llm._client
        llm._client = None
        try:
            llm.get_client()
            client_cls.assert_called_with(**sentinel)
        finally:
            llm._client = original
