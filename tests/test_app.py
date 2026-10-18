"""Headless runs of the Streamlit app (no network, no Gemini)."""

from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from job_agent.schemas import JobDescription, Seniority  # noqa: E402
from job_agent.tracker_db import TrackerRepo, get_engine  # noqa: E402
from job_agent.tracker_logic import ApplicationInput, Status, today_local  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
VISITOR = "c" * 32

FAKE_JD = JobDescription(
    job_title="Data Science Intern", company="Acme", seniority=Seniority.INTERN,
    required_skills=["python", "sql"], preferred_skills=["tableau", "docker"],
    responsibilities=[], keywords=["python"], summary="x",
)


@pytest.fixture(autouse=True)
def fresh_app(monkeypatch, tmp_path):
    import streamlit as st

    st.cache_resource.clear()
    monkeypatch.setenv("DEMO_SESSION_LIMIT", "1")
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "50")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'tracker.db'}")
    monkeypatch.setattr("job_agent.jd_parser.parse_jd", lambda text, *a, **k: FAKE_JD)
    yield
    st.cache_resource.clear()


def new_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["visitor_id"] = VISITOR
    return at


def test_demo_flow_shows_score_then_enforces_session_limit():
    at = new_app().run()
    assert not at.exception

    at.button(key="sample_btn").click().run()
    assert "Acme Analytics" in at.text_area(key="jd_text").value

    at.button(key="analyze_btn").click().run()
    assert not at.exception
    assert any("/100" in m.value for m in at.metric)
    assert any("runs left this session: 0" in c.value for c in at.sidebar.caption)  # not stale

    at.button(key="analyze_btn").click().run()  # second run exceeds DEMO_SESSION_LIMIT=1
    assert any("limit reached" in w.value for w in at.warning)


def test_short_jd_is_rejected():
    at = new_app().run()
    at.text_area(key="jd_text").set_value("too short")
    at.button(key="analyze_btn").click().run()
    assert any("longer job description" in w.value for w in at.warning)
    assert not at.metric  # an empty tracker shows no metrics either


def test_tracker_starts_empty_without_errors():
    at = new_app().run()
    assert not at.exception
    assert any("No applications yet" in i.value for i in at.info)


def test_save_analysis_to_tracker_then_it_appears_in_the_tracker():
    at = new_app().run()
    at.button(key="sample_btn").click().run()
    at.button(key="analyze_btn").click().run()

    rid = at.session_state["result_id"]
    at.button(key=f"sv_btn_{rid}").click().run()
    assert not at.exception

    saved = TrackerRepo(get_engine()).list_applications(VISITOR)
    assert len(saved) == 1
    app = saved[0]
    assert (app.company, app.role, app.status) == ("Acme", "Data Science Intern", Status.SAVED)
    assert app.match_score is not None and app.resume_version == "Sample resume"
    assert app.jd_text  # the JD text is kept for the later tailoring step

    assert not any("No applications yet" in i.value for i in at.info)  # tracker now lists it


def test_due_followup_shows_banner_and_can_be_snoozed():
    repo = TrackerRepo(get_engine())
    today = today_local()
    app = repo.add(VISITOR, ApplicationInput(company="LateCo", role="Analyst", status=Status.APPLIED,
                                             applied_on=today - timedelta(days=10)), today)
    at = new_app().run()
    assert not at.exception
    assert any("follow-up(s) due" in w.value for w in at.warning)

    at.button(key=f"fu_{app.id}").click().run()
    assert not at.exception
    assert repo.get(VISITOR, app.id).follow_up_count == 1
    assert not any("follow-up(s) due" in w.value for w in at.warning)


def test_other_visitors_do_not_see_my_applications():
    repo = TrackerRepo(get_engine())
    repo.add(VISITOR, ApplicationInput(company="Mine", role="x", status=Status.APPLIED,
                                       applied_on=today_local() - timedelta(days=30)))
    other = AppTest.from_file(APP_PATH, default_timeout=30)
    other.session_state["visitor_id"] = "d" * 32
    other.run()
    assert not other.exception
    assert not any("follow-up(s) due" in w.value for w in other.warning)
    assert any("No applications yet" in i.value for i in other.info)
