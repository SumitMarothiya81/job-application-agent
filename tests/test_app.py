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
    responsibilities=[], keywords=["python", "result"], summary="x",
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


# ---------------------------------------------------------------- Discover tab
from job_agent.discovery import JobListing, SearchResult, listing_id  # noqa: E402

FOUND = [
    JobListing(source="adzuna", external_id="1", title="Python Data Intern", company="Acme", location="Pune",
               description="Python, SQL and pandas. Freshers welcome.", url="https://jobs.example.com/1"),
    JobListing(source="adzuna", external_id="2", title="Senior Data Architect", company="BigCo",
               description="8+ years of experience", url="https://jobs.example.com/2"),
]


@pytest.fixture
def fake_search(monkeypatch):
    calls = []

    def search(self, what, where=None, page=1, results_per_page=20, max_days_old=14, sort_by="date"):
        calls.append((self._app_id, what))
        return SearchResult(list(FOUND), False, 42)

    monkeypatch.setattr("job_agent.discovery.AdzunaClient.search", search)
    return calls


def test_discover_without_keys_explains_instead_of_crashing(monkeypatch):
    monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
    monkeypatch.delenv("ADZUNA_APP_KEY", raising=False)
    at = new_app().run()
    at.button(key="disc_search_btn").click().run()
    assert not at.exception
    assert any("isn't configured" in w.value for w in at.warning)


def test_discover_search_ranks_hides_experienced_and_saves_to_tracker(monkeypatch, fake_search):
    monkeypatch.setenv("ADZUNA_APP_ID", "server-id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "server-key")
    at = new_app().run()
    at.button(key="disc_search_btn").click().run()
    assert not at.exception and fake_search == [("server-id", "data science intern")]

    shown = [m.value for m in at.markdown]
    assert any("1 job(s) shown" in text and "1 experienced role(s) hidden" in text for text in shown)
    assert any("Python Data Intern" in text for text in shown)
    assert not any("Senior Data Architect" in text for text in shown)

    at.button(key=f"disc_save_{listing_id(FOUND[0])}").click().run()
    assert not at.exception
    saved = TrackerRepo(get_engine()).list_applications(VISITOR)
    assert len(saved) == 1
    assert (saved[0].company, saved[0].role, saved[0].status) == ("Acme", "Python Data Intern", Status.SAVED)
    assert saved[0].source_url == "https://jobs.example.com/1" and "Quick fit estimate" in saved[0].notes
    assert any("1 already in your tracker" in m.value for m in at.markdown)   # now hidden from the results


def test_discover_demo_search_limit_and_own_keys_bypass(monkeypatch, fake_search):
    monkeypatch.setenv("ADZUNA_APP_ID", "server-id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "server-key")
    monkeypatch.setenv("DISCOVERY_SESSION_LIMIT", "1")
    at = new_app().run()
    at.button(key="disc_search_btn").click().run()
    at.button(key="disc_search_btn").click().run()          # second call exceeds the demo limit
    assert any("Demo search limit reached" in w.value for w in at.warning)
    assert len(fake_search) == 1

    at.text_input(key="adz_id").set_value("my-own-id")
    at.text_input(key="adz_key").set_value("my-own-key")
    at.button(key="disc_search_btn").click().run()          # own keys: not limited, uses own id
    assert not at.exception and fake_search[-1][0] == "my-own-id"


def test_discover_loads_a_job_into_the_match_tab(monkeypatch, fake_search):
    monkeypatch.setenv("ADZUNA_APP_ID", "server-id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "server-key")
    at = new_app().run()
    at.button(key="disc_search_btn").click().run()
    at.button(key=f"disc_an_{listing_id(FOUND[0])}").click().run()
    assert not at.exception
    assert "Python Data Intern - Acme" in at.text_area(key="jd_text").value


# ---------------------------------------------------------------- Apply assistant tab
from job_agent import agent as agent_module  # noqa: E402
from job_agent.agent import AgentDeps, EmailDraft, JudgedBullet, JudgeOutput, TailorOutput, TailoredBullet  # noqa: E402

ASSISTANT_JD = "Data Science Intern at Acme. We need Python, SQL, pandas and scikit-learn. Six month internship in Pune."


@pytest.fixture
def fake_agent(monkeypatch):
    calls = {"tailor": 0, "email": 0}

    def tailor(payload):
        calls["tailor"] += 1
        return TailorOutput(items=[TailoredBullet(id=b["id"], rewritten="Key result: " + b["original"])
                                   for b in payload["bullets"]])

    def email(payload):
        calls["email"] += 1
        return EmailDraft(subject="Application: Data Science Intern",
                          body="Hi [Recruiter name],\n\nI use Python and SQL.\n\nThanks,\n[Your name]")

    deps = AgentDeps(
        parse_jd=lambda text: FAKE_JD.model_copy(update={"required_skills": ["python", "sql", "pandas"]}),
        tailor=tailor,
        judge=lambda payload: JudgeOutput(items=[JudgedBullet(id=i["id"], supported=True) for i in payload["items"]]),
        email=email)
    monkeypatch.setattr(agent_module, "default_deps", lambda: deps)
    return calls


def start_assistant(at):
    at.text_area(key="agent_jd").set_value(ASSISTANT_JD)
    at.button(key="agent_start").click().run()
    return at


def test_assistant_full_flow_in_the_browser(fake_agent):
    at = start_assistant(new_app().run())
    assert not at.exception
    assert any("Step 1 of 2" in m.value for m in at.markdown)                # paused for the bullet review
    assert fake_agent == {"tailor": 1, "email": 0}

    at.button(key="agent_bullets_submit").click().run()                      # accept every suggestion (default)
    assert not at.exception and any("Step 2 of 2" in m.value for m in at.markdown)
    assert at.text_input(key="agent_email_subject").value.startswith("Application")

    at.text_area(key="agent_email_body").set_value("Hi [Recruiter name],\nI use Python.\n[Your name]")
    at.button(key="agent_email_approve").click().run()
    assert not at.exception
    assert any("application kit is ready" in s.value for s in at.success)
    assert any("Nothing has been sent" in s.value for s in at.success)

    at.button(key="agent_sv_btn").click().run()                              # save to the tracker
    saved = TrackerRepo(get_engine()).list_applications(VISITOR)
    assert len(saved) == 1 and saved[0].company == "Acme" and saved[0].status == Status.SAVED
    assert saved[0].resume_version.startswith("Tailored: Acme") and saved[0].jd_text == ASSISTANT_JD
    assert "Tailored" in saved[0].notes and "email drafted" in saved[0].notes


def test_assistant_unchecked_rewrites_are_not_used_and_email_can_be_skipped(fake_agent):
    at = start_assistant(new_app().run())
    for checkbox in at.checkbox:
        if checkbox.key and checkbox.key.startswith("agent_acc_"):
            checkbox.uncheck()
    at.button(key="agent_bullets_submit").click().run()
    at.button(key="agent_email_skip").click().run()
    assert not at.exception
    assert any("No rewrites were accepted" in m.value for m in at.markdown)
    assert any("No email was kept" in m.value for m in at.markdown)


def test_assistant_start_over_returns_to_the_start(fake_agent):
    at = start_assistant(new_app().run())
    at.button(key="agent_reset").click().run()
    assert not at.exception and at.button(key="agent_start") is not None


def test_assistant_validation_and_demo_limit(fake_agent, monkeypatch):
    monkeypatch.setenv("AGENT_SESSION_LIMIT", "1")
    at = new_app().run()
    at.text_area(key="agent_jd").set_value("too short")
    at.button(key="agent_start").click().run()
    assert any("longer job description" in w.value for w in at.warning)

    at = start_assistant(new_app().run())                                    # uses the one allowed run
    at.button(key="agent_reset").click().run()
    at.text_area(key="agent_jd").set_value(ASSISTANT_JD)
    at.button(key="agent_start").click().run()
    assert any("Demo assistant limit reached" in w.value for w in at.warning)
    assert fake_agent["tailor"] == 1                                         # the second run never started


def test_assistant_shows_a_friendly_error_when_a_model_call_fails(monkeypatch):
    def boom(payload):
        raise RuntimeError("API key not valid. Please pass a valid API key.")

    deps = AgentDeps(parse_jd=lambda text: FAKE_JD, tailor=boom, judge=None,
                     email=lambda payload: EmailDraft(subject="s", body="b"))
    monkeypatch.setattr(agent_module, "default_deps", lambda: deps)
    at = start_assistant(new_app().run())
    assert not at.exception
    assert any("rejected that API key" in e.value for e in at.error)
    assert at.button(key="agent_start") is not None                           # back at the start, can retry


def test_assistant_copies_the_job_from_the_match_tab(fake_agent):
    at = new_app().run()
    at.text_area(key="jd_text").set_value(ASSISTANT_JD)
    at.button(key="agent_copy").click().run()
    assert at.text_area(key="agent_jd").value == ASSISTANT_JD


def test_assistant_says_which_resume_it_uses(fake_agent):
    at = new_app().run()
    assert any("Resume in use" in c.value and "Sample resume" in c.value for c in at.caption)
