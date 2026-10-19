from datetime import date, datetime, timedelta

import pytest
from pydantic import ValidationError

from job_agent import tracker_logic as t
from job_agent.tracker_logic import ApplicationInput, ApplicationView, Status, StatusChangeView


def make_view(**overrides) -> ApplicationView:
    base = dict(id=1, company="Acme", role="Data Intern", status=Status.APPLIED,
                applied_on=date(2026, 10, 10), created_at=datetime(2026, 10, 10), history=[])
    base.update(overrides)
    return ApplicationView(**base)


def change(to, frm=None):
    return StatusChangeView(from_status=frm, to_status=to, changed_at=datetime(2026, 10, 11))


# ---- scheduling -------------------------------------------------------------
def test_schedule_follow_up():
    day = date(2026, 10, 18)
    assert t.schedule_follow_up(Status.APPLIED, day) == date(2026, 10, 25)
    assert t.schedule_follow_up(Status.INTERVIEW, day) == date(2026, 10, 21)
    assert t.schedule_follow_up(Status.ONLINE_ASSESSMENT, day) == date(2026, 10, 21)
    for status in (Status.SAVED, Status.OFFER, Status.REJECTED, Status.GHOSTED, Status.WITHDRAWN):
        assert t.schedule_follow_up(status, day) is None


def test_today_local_falls_back_for_unknown_timezone():
    assert isinstance(t.today_local("Not/AZone"), date)
    assert isinstance(t.today_local("Asia/Kolkata"), date)


# ---- validation ---------------------------------------------------------------
def test_input_cleans_blank_strings_and_whitespace():
    data = ApplicationInput(company="  Acme  ", role="DS", source_url="  ", notes="", contact_email="")
    assert data.company == "Acme" and data.source_url is None and data.notes is None
    assert data.contact_email is None


@pytest.mark.parametrize("bad", [
    dict(company="", role="x"),
    dict(company="x", role=""),
    dict(company="x", role="y", source_url="ftp://example.com"),
    dict(company="x", role="y", source_url="http://exa mple.com"),
    dict(company="x", role="y", contact_email="not-an-email"),
    dict(company="x", role="y", match_score=101),
    dict(company="x", role="y", match_score=-1),
    dict(company="x" * 121, role="y"),
    dict(company="x", role="y", status="flying"),
])
def test_input_rejects_bad_values(bad):
    with pytest.raises(ValidationError):
        ApplicationInput(**bad)


def test_input_accepts_valid_url_and_email():
    data = ApplicationInput(company="x", role="y", source_url="https://jobs.example.com/1",
                            contact_email="hr@example.com", match_score=0)
    assert data.match_score == 0


# ---- stats --------------------------------------------------------------------
def test_stats_rates_and_resume_versions():
    apps = [
        make_view(id=1, status=Status.INTERVIEW, resume_version="v1",
                  history=[change(Status.APPLIED), change(Status.INTERVIEW, Status.APPLIED)]),
        make_view(id=2, status=Status.REJECTED, resume_version="v1", history=[change(Status.REJECTED)]),
        make_view(id=3, status=Status.APPLIED, resume_version="v2"),
        make_view(id=4, status=Status.GHOSTED, resume_version="v2"),
        make_view(id=5, status=Status.SAVED, applied_on=None),           # never applied
    ]
    stats = t.compute_stats(apps, date(2026, 10, 18))
    assert stats["total"] == 5 and stats["applied"] == 4
    assert stats["responded"] == 2 and stats["response_rate"] == 0.5
    assert stats["interviewed"] == 1 and stats["interview_rate"] == 0.25
    assert stats["by_resume_version"]["v1"] == {"applied": 2, "responded": 2, "response_rate": 1.0}
    assert stats["by_resume_version"]["v2"]["response_rate"] == 0.0
    assert stats["by_status"][Status.SAVED] == 1


def test_stats_with_no_applications():
    stats = t.compute_stats([], date(2026, 10, 18))
    assert stats["response_rate"] is None and stats["interview_rate"] is None
    assert len(stats["weekly"]) == 8


def test_stats_weekly_counts_use_monday_weeks():
    today = date(2026, 10, 18)  # a Sunday: its week starts Monday 12 Oct
    apps = [make_view(id=1, applied_on=date(2026, 10, 12)), make_view(id=2, applied_on=date(2026, 10, 18)),
            make_view(id=3, applied_on=date(2026, 10, 11))]
    weekly = t.compute_stats(apps, today)["weekly"]
    assert weekly[-1] == {"week_start": date(2026, 10, 12), "applications": 2}
    assert weekly[-2]["applications"] == 1


# ---- calendar -----------------------------------------------------------------
def test_ics_content_and_escaping():
    app = make_view(company="Acme, Inc; Labs", next_follow_up=date(2026, 10, 25),
                    contact_name="Riya", contact_email="hr@example.com")
    text = t.follow_up_ics(app, now=datetime(2026, 10, 18, 12, 0, 0))
    assert text.startswith("BEGIN:VCALENDAR\r\n") and text.endswith("END:VCALENDAR\r\n")
    assert "DTSTART;VALUE=DATE:20261025" in text and "DTEND;VALUE=DATE:20261026" in text
    assert "DTSTAMP:20261018T120000Z" in text
    assert "Acme\\, Inc\\; Labs" in text              # commas and semicolons escaped
    assert "\n" not in text.replace("\r\n", "")       # only CRLF line endings
    assert all(len(line) <= 75 for line in text.split("\r\n"))


def test_ics_requires_a_follow_up_date():
    with pytest.raises(ValueError):
        t.follow_up_ics(make_view(next_follow_up=None))


def test_export_model_round_trip():
    export = t.TrackerExport(exported_on=date(2026, 10, 18), applications=[
        t.ExportedApplication(company="A", role="B", status=Status.APPLIED,
                              history=[t.ExportedChange(to_status=Status.APPLIED, changed_at=datetime(2026, 10, 1))])])
    again = t.TrackerExport.model_validate_json(export.model_dump_json())
    assert again.applications[0].history[0].to_status == Status.APPLIED
    with pytest.raises(ValidationError):
        t.TrackerExport.model_validate_json('{"version": 2, "exported_on": "2026-10-18", "applications": []}')


def test_to_local_converts_utc_to_ist_and_survives_bad_zone():
    utc = datetime(2026, 10, 18, 6, 51)
    assert t.to_local(utc, "Asia/Kolkata") == datetime(2026, 10, 18, 12, 21, tzinfo=t.to_local(utc, "Asia/Kolkata").tzinfo)
    assert t.to_local(utc, "Asia/Kolkata").hour == 12 and t.to_local(utc, "Asia/Kolkata").minute == 21
    assert t.to_local(utc, "Not/AZone") == utc

