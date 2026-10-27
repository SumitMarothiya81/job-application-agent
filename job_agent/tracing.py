"""Optional run tracing (LangSmith). OFF unless you switch it on; metadata-only unless you allow content.

Why this matters here: a trace of an assistant run contains the prompts, i.e. job text and resume bullets.
So there are three modes, decided once at start-up:

  off       (default) every LangChain/LangSmith tracing switch is forced to false, even a stray
            LANGSMITH_TRACING=true left in your shell or another project's .env
  metadata  ENABLE_TRACING=true + LANGSMITH_API_KEY: step names, timings, model names, retries are sent;
            prompts and answers are HIDDEN
  full      additionally TRACE_CONTENT=true: prompts and answers are sent too (use for your own debugging only)
"""

import os

_TRUE = {"1", "true", "yes", "on"}
_ALL_TRACING_SWITCHES = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
_HIDE_SWITCHES = ("LANGSMITH_HIDE_INPUTS", "LANGSMITH_HIDE_OUTPUTS", "LANGCHAIN_HIDE_INPUTS", "LANGCHAIN_HIDE_OUTPUTS")


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE


def configure_tracing() -> str:
    """Set the environment for the chosen mode and return it: 'off', 'metadata' or 'full'.

    Must run before the first LangChain call, because LangSmith reads these settings once and caches them.
    """
    if not (_flag("ENABLE_TRACING") and os.getenv("LANGSMITH_API_KEY")):
        for name in _ALL_TRACING_SWITCHES:
            os.environ[name] = "false"
        return "off"

    full = _flag("TRACE_CONTENT")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "job-application-agent")
    for name in _HIDE_SWITCHES:
        os.environ[name] = "false" if full else "true"
    return "full" if full else "metadata"


def notice(mode: str) -> str | None:
    """A line for the sidebar so visitors know what this server records (None when tracing is off)."""
    if mode == "metadata":
        return ("This server records run traces to LangSmith: step names, timings and model names. "
                "Your resume and job text are NOT included.")
    if mode == "full":
        return ("This server records run traces to LangSmith INCLUDING the text you submit "
                "(job description and resume bullets).")
    return None
