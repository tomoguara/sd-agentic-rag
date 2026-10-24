"""JSON calls to the chat model for the Planner, Critic and Self-Review.

`json_call()` is the only way the agent's LLM-driven components talk to the
model: temperature 0, JSON response format, parsed and validated against a
small schema, and every call (successful or not) counted on the budget.
Invalid output raises LLMCallError so the caller can fall back to its
deterministic path.
"""

import json
import re
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from sd_agentic_rag.agent.budget import BudgetTracker


# Schema: required key -> expected type(s)
Schema = Dict[str, Union[Type, Tuple[Type, ...]]]


class LLMCallError(Exception):
    """The model call failed or returned output that is not valid for the schema."""


_client = None


def get_client():
    """The shared OpenAI-compatible client, built from model_selector.get_client_kwargs()."""
    global _client
    if _client is None:
        import openai
        from sd_agentic_rag.model_selector import get_client_kwargs
        _client = openai.OpenAI(**get_client_kwargs())
    return _client


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object from model output, tolerating think tags and code fences."""
    if not text:
        raise LLMCallError("empty response")
    cleaned = _THINK_RE.sub("", text).strip()
    fence = _FENCE_RE.match(cleaned)
    if fence:
        cleaned = fence.group(1)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some local models wrap the object in prose; take the outermost braces
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMCallError(f"response is not JSON: {cleaned[:80]!r}")
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as e:
            raise LLMCallError(f"response is not JSON: {e}")
    if not isinstance(value, dict):
        raise LLMCallError("response is not a JSON object")
    return value


def validate(value: Dict[str, Any], schema: Schema) -> Dict[str, Any]:
    """Check required keys and their types; raise LLMCallError on a violation."""
    for key, expected in schema.items():
        if key not in value:
            raise LLMCallError(f"missing key {key!r}")
        types = expected if isinstance(expected, tuple) else (expected,)
        # bool is an int subclass; only accept it where bool is expected
        if isinstance(value[key], bool) and bool not in types:
            raise LLMCallError(f"key {key!r} has type bool")
        if not isinstance(value[key], types):
            raise LLMCallError(f"key {key!r} has type {type(value[key]).__name__}")
    return value


def json_call(
    system: str,
    user: str,
    schema: Schema,
    budget: BudgetTracker,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    max_tokens: int = 800,
) -> Dict[str, Any]:
    """Ask the chat model for a JSON object and validate it.

    Raises:
        BudgetExceeded: Before the call, if the budget does not allow it
        LLMCallError: If the call fails or the output is invalid
    """
    budget.check_llm_call()
    if model is None:
        from sd_agentic_rag.model_selector import get_llm_model_name
        model = get_llm_model_name()
    client = client or get_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        budget.record_llm_call(0)
        raise LLMCallError(f"LLM call failed: {e}") from e

    usage = getattr(response, "usage", None)
    budget.record_llm_call(getattr(usage, "total_tokens", 0) or 0)
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        raise LLMCallError(f"malformed response: {e}") from e
    return validate(parse_json(content), schema)


# Signature the components accept, so tests can inject a stub
JsonCaller = Callable[..., Dict[str, Any]]
