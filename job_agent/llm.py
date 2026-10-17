"""One place that creates the LLM client, with model fallback and per-request API keys."""

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from job_agent.config import GEMINI_MODELS, require_api_key

log = logging.getLogger(__name__)

# Error text that means "try again / try another model", not "your code is wrong".
_TRANSIENT_MARKERS = (
    "503", "UNAVAILABLE", "high demand", "overloaded",
    "429", "RESOURCE_EXHAUSTED", "quota",
    "500", "504", "DEADLINE_EXCEEDED", "timed out",
)

# Bring-your-own-key: inside `with use_api_key(key):` every Gemini call uses that key.
_api_key_override: ContextVar[str | None] = ContextVar("api_key_override", default=None)


@contextmanager
def use_api_key(api_key: str | None):
    """Use `api_key` for all Gemini calls inside the block (None = the server's key)."""
    token = _api_key_override.set(api_key.strip() if api_key else None)
    try:
        yield
    finally:
        _api_key_override.reset(token)


def get_llm(model: str | None = None, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Return a Gemini chat model for the current key (override, else server key)."""
    api_key = _api_key_override.get() or require_api_key()
    return _build_llm(model or GEMINI_MODELS[0], temperature, api_key)


@lru_cache(maxsize=8)
def _build_llm(model: str, temperature: float, api_key: str) -> ChatGoogleGenerativeAI:
    """Gemini 3 models work best with their default temperature, so we only
    set it for older models."""
    kwargs = {}
    if not model.startswith("gemini-3"):
        kwargs["temperature"] = temperature
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        max_retries=2,  # retry briefly, then let our fallback switch models
        **kwargs,
    )


def is_transient_error(error: Exception) -> bool:
    text = str(error)
    return any(marker.lower() in text.lower() for marker in _TRANSIENT_MARKERS)


def invoke_with_fallback(messages, schema=None):
    """Call Gemini; on overload/rate-limit errors, try the next configured model.

    schema: optional Pydantic class for structured output.
    Non-transient errors (bad key, bad request) are raised immediately.
    """
    last_error: Exception | None = None
    for model in GEMINI_MODELS:
        try:
            llm = get_llm(model)
            if schema is not None:
                llm = llm.with_structured_output(schema)
            result = llm.invoke(messages)
            log.info("Model used: %s", model)
            return result
        except Exception as error:  # noqa: BLE001
            if not is_transient_error(error):
                raise
            log.warning("Model %s unavailable (%s). Trying next...", model, str(error)[:80])
            last_error = error
    raise RuntimeError(
        f"All configured models failed: {GEMINI_MODELS}. Try again in a few minutes."
    ) from last_error
