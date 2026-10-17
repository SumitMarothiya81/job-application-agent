import pytest

from job_agent import llm


class FakeLLM:
    def __init__(self, error=None, content="ok"):
        self.error, self.content = error, content

    def invoke(self, messages):
        if self.error:
            raise self.error
        return self.content


def test_is_transient_error():
    assert llm.is_transient_error(Exception("503 UNAVAILABLE high demand"))
    assert llm.is_transient_error(Exception("429 RESOURCE_EXHAUSTED"))
    assert not llm.is_transient_error(Exception("400 API key not valid"))


def test_falls_back_to_next_model(monkeypatch):
    monkeypatch.setattr(llm, "GEMINI_MODELS", ["busy", "good"])
    fakes = {"busy": FakeLLM(Exception("503 UNAVAILABLE")), "good": FakeLLM(content="hello")}
    monkeypatch.setattr(llm, "get_llm", lambda model=None, temperature=0.0: fakes[model])
    assert llm.invoke_with_fallback("hi") == "hello"


def test_non_transient_error_is_raised_immediately(monkeypatch):
    monkeypatch.setattr(llm, "GEMINI_MODELS", ["a", "b"])
    fakes = {"a": FakeLLM(Exception("400 API key not valid")), "b": FakeLLM(content="x")}
    monkeypatch.setattr(llm, "get_llm", lambda model=None, temperature=0.0: fakes[model])
    with pytest.raises(Exception, match="API key not valid"):
        llm.invoke_with_fallback("hi")


def test_all_models_failing_gives_clear_error(monkeypatch):
    monkeypatch.setattr(llm, "GEMINI_MODELS", ["a", "b"])
    fakes = {m: FakeLLM(Exception("503 UNAVAILABLE")) for m in "ab"}
    monkeypatch.setattr(llm, "get_llm", lambda model=None, temperature=0.0: fakes[model])
    with pytest.raises(RuntimeError, match="All configured models failed"):
        llm.invoke_with_fallback("hi")
