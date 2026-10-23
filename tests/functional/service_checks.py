"""Shared skip logic for functional tests that need live services.

A functional test may skip only when a service it needs (Qdrant, Ollama,
OpenAI) is genuinely unreachable. Any other error during setup - an import
error, a bad argument, a dimension mismatch - is a real failure and must not
be reported as a skip.
"""
import pytest


def is_connection_error(exc: BaseException) -> bool:
    """True if exc, or anything in its cause/context chain, is a connection failure."""
    connection_types = [ConnectionError, TimeoutError]
    try:
        import httpx
        connection_types += [httpx.ConnectError, httpx.ConnectTimeout]
    except ImportError:
        pass
    try:
        import openai
        connection_types += [openai.APIConnectionError]
    except ImportError:
        pass

    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, tuple(connection_types)):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def skip_if_unavailable(exc: BaseException, service: str = "Qdrant") -> None:
    """Skip the current test if exc means `service` is unreachable; otherwise re-raise it.

    Call from inside an `except` block:

        except Exception as e:
            skip_if_unavailable(e, "Qdrant")
    """
    if is_connection_error(exc):
        pytest.skip(f"{service} not available: {exc}")
    raise exc
