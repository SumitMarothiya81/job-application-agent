import os

import pytest
from langsmith import utils as ls_utils

from job_agent import tracing

NAMES = ("ENABLE_TRACING", "TRACE_CONTENT", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", *tracing._ALL_TRACING_SWITCHES,
         *tracing._HIDE_SWITCHES)


@pytest.fixture(autouse=True)
def clean_environment():
    """configure_tracing() writes to os.environ directly, so restore everything afterwards."""
    before = dict(os.environ)
    for name in NAMES:
        os.environ.pop(name, None)
    ls_utils.get_env_var.cache_clear()
    yield
    os.environ.clear()
    os.environ.update(before)
    ls_utils.get_env_var.cache_clear()


def enabled() -> bool:
    ls_utils.get_env_var.cache_clear()
    return bool(ls_utils.tracing_is_enabled())


def test_default_is_off_even_with_stray_switches_in_the_environment():
    os.environ.update(LANGSMITH_TRACING="true", LANGCHAIN_TRACING_V2="true")      # e.g. left over in your shell
    assert tracing.configure_tracing() == "off" and not enabled()


def test_a_key_alone_or_the_switch_alone_does_not_turn_it_on():
    os.environ["LANGSMITH_API_KEY"] = "k"
    assert tracing.configure_tracing() == "off" and not enabled()
    os.environ.pop("LANGSMITH_API_KEY")
    os.environ["ENABLE_TRACING"] = "true"
    assert tracing.configure_tracing() == "off" and not enabled()


def test_metadata_mode_hides_prompts_and_answers():
    os.environ.update(ENABLE_TRACING="true", LANGSMITH_API_KEY="k")
    assert tracing.configure_tracing() == "metadata" and enabled()
    assert all(os.environ[n] == "true" for n in tracing._HIDE_SWITCHES)
    assert os.environ["LANGSMITH_PROJECT"] == "job-application-agent"


def test_full_mode_needs_an_explicit_content_switch_and_respects_a_custom_project():
    os.environ.update(ENABLE_TRACING="true", LANGSMITH_API_KEY="k", TRACE_CONTENT="true", LANGSMITH_PROJECT="mine")
    assert tracing.configure_tracing() == "full" and enabled()
    assert all(os.environ[n] == "false" for n in tracing._HIDE_SWITCHES) and os.environ["LANGSMITH_PROJECT"] == "mine"


def test_visitor_notice_matches_the_mode():
    assert tracing.notice("off") is None
    assert "NOT included" in tracing.notice("metadata")
    assert "INCLUDING the text you submit" in tracing.notice("full")
