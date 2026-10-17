"""Headless run of the Streamlit app (no network, no Gemini)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from job_agent.schemas import JobDescription, Seniority  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

FAKE_JD = JobDescription(
    job_title="Data Science Intern", company="Acme", seniority=Seniority.INTERN,
    required_skills=["python", "sql"], preferred_skills=["tableau", "docker"],
    responsibilities=[], keywords=["python"], summary="x",
)


@pytest.fixture(autouse=True)
def fresh_app(monkeypatch):
    import streamlit as st

    st.cache_resource.clear()
    monkeypatch.setenv("DEMO_SESSION_LIMIT", "1")
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "50")
    monkeypatch.setattr("job_agent.jd_parser.parse_jd", lambda text, *a, **k: FAKE_JD)
    yield
    st.cache_resource.clear()


def test_demo_flow_shows_score_then_enforces_session_limit():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception

    at.button(key="sample_btn").click().run()
    assert "Acme Analytics" in at.text_area(key="jd_text").value

    at.button(key="analyze_btn").click().run()
    assert not at.exception
    assert any("/100" in m.value for m in at.metric)

    at.button(key="analyze_btn").click().run()  # second run exceeds DEMO_SESSION_LIMIT=1
    assert any("limit reached" in w.value for w in at.warning)


def test_short_jd_is_rejected():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.text_area(key="jd_text").set_value("too short")
    at.button(key="analyze_btn").click().run()
    assert any("longer job description" in w.value for w in at.warning)
    assert not at.metric
