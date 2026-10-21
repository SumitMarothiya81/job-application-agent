from job_agent import llm
from job_agent.limits import UsageLimiter


def test_session_limit():
    limiter = UsageLimiter(per_session=2, per_day=100)
    assert limiter.try_consume("a")[0] and limiter.try_consume("a")[0]
    allowed, message = limiter.try_consume("a")
    assert not allowed and "session" in message
    assert limiter.try_consume("b")[0]  # other sessions unaffected


def test_daily_limit_and_units():
    limiter = UsageLimiter(per_session=10, per_day=3)
    assert limiter.try_consume("a", units=2)[0]
    assert not limiter.try_consume("b", units=2)[0]  # would exceed 3 for the day
    assert limiter.try_consume("b", units=1)[0]
    allowed, message = limiter.try_consume("c")
    assert not allowed and "daily" in message


def test_limits_reset_on_new_day():
    day = {"value": "2026-10-18"}
    limiter = UsageLimiter(per_session=1, per_day=1, today=lambda: day["value"])
    assert limiter.try_consume("a")[0] and not limiter.try_consume("a")[0]
    day["value"] = "2026-10-19"
    assert limiter.try_consume("a")[0]


def test_remaining():
    limiter = UsageLimiter(per_session=5, per_day=3)
    limiter.try_consume("a")
    assert limiter.remaining("a") == 2  # day cap (3-1) is tighter than session cap (5-1)


def test_use_api_key_overrides_server_key(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "require_api_key", lambda: "server-key")
    monkeypatch.setattr(llm, "_build_llm", lambda model, temp, key: seen.append(key) or object())

    llm.get_llm("m")
    with llm.use_api_key("  user-key  "):
        llm.get_llm("m")
    llm.get_llm("m")
    assert seen == ["server-key", "user-key", "server-key"]  # trimmed, and restored afterwards


def test_gemini_calls_have_a_time_limit_that_can_be_changed(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "ChatGoogleGenerativeAI", lambda **kwargs: seen.append(kwargs) or object())
    monkeypatch.setattr(llm, "require_api_key", lambda: "k")
    llm._build_llm.cache_clear()
    llm.get_llm("gemini-3.5-flash")
    monkeypatch.setenv("GEMINI_TIMEOUT", "25")
    llm._build_llm.cache_clear()
    llm.get_llm("gemini-3.7-flash")
    llm._build_llm.cache_clear()
    assert [kw["timeout"] for kw in seen] == [60.0, 25.0]


def test_speed_knobs_are_opt_in(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "ChatGoogleGenerativeAI", lambda **kwargs: seen.append(kwargs) or object())
    monkeypatch.setattr(llm, "require_api_key", lambda: "k")
    for name in ("GEMINI_THINKING_BUDGET", "GEMINI_MAX_RETRIES"):
        monkeypatch.delenv(name, raising=False)
    llm._build_llm.cache_clear()
    llm.get_llm("gemini-3.5-flash")
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "0")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "1")
    llm._build_llm.cache_clear()
    llm.get_llm("gemini-3.7-flash")
    llm._build_llm.cache_clear()
    assert "thinking_budget" not in seen[0] and seen[0]["max_retries"] == 2          # defaults unchanged
    assert seen[1]["thinking_budget"] == 0 and seen[1]["max_retries"] == 1
