import json
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from job_agent import tracker_db
from job_agent.tracker_db import LimitError, NotFoundError, TrackerError, TrackerRepo, get_engine
from job_agent.tracker_logic import ApplicationInput, Status

TODAY = date(2026, 10, 18)
ALICE, BOB = "a" * 32, "b" * 32


@pytest.fixture
def repo():
    return TrackerRepo(get_engine("sqlite:///:memory:"))


def new(repo, user=ALICE, status=Status.SAVED, **kw):
    data = ApplicationInput(company=kw.pop("company", "Acme"), role=kw.pop("role", "Data Intern"),
                            status=status, **kw)
    return repo.add(user, data, today=TODAY)


# ---- create ---------------------------------------------------------------------
def test_add_saved_has_no_dates(repo):
    app = new(repo)
    assert app.applied_on is None and app.next_follow_up is None
    assert [h.to_status for h in app.history] == [Status.SAVED]


def test_add_applied_sets_date_and_follow_up(repo):
    app = new(repo, status=Status.APPLIED)
    assert app.applied_on == TODAY and app.next_follow_up == TODAY + timedelta(days=7)


def test_add_applied_in_the_past_is_already_due(repo):
    app = new(repo, status=Status.APPLIED, applied_on=TODAY - timedelta(days=10))
    assert app.next_follow_up == TODAY - timedelta(days=3)
    assert [a.id for a in repo.due_reminders(ALICE, TODAY)] == [app.id]


def test_add_terminal_status_never_has_follow_up(repo):
    app = new(repo, status=Status.REJECTED, next_follow_up=TODAY)
    assert app.next_follow_up is None


def test_add_rejects_bad_user_id(repo):
    with pytest.raises(ValueError):
        repo.add("../../etc", ApplicationInput(company="x", role="y"))


# ---- isolation ---------------------------------------------------------------------
def test_users_cannot_touch_each_others_applications(repo):
    mine = new(repo, user=ALICE)
    assert repo.list_applications(BOB) == []
    for call in (lambda: repo.get(BOB, mine.id),
                 lambda: repo.change_status(BOB, mine.id, Status.APPLIED),
                 lambda: repo.update_fields(BOB, mine.id, {"notes": "hacked"}),
                 lambda: repo.mark_followed_up(BOB, mine.id),
                 lambda: repo.delete(BOB, mine.id)):
        with pytest.raises(NotFoundError):
            call()
    assert repo.get(ALICE, mine.id).notes is None
    assert repo.delete_all(BOB) == 0 and len(repo.list_applications(ALICE)) == 1


# ---- status changes ---------------------------------------------------------------
def test_status_flow_and_history(repo):
    app = new(repo)
    app = repo.change_status(ALICE, app.id, Status.APPLIED, today=TODAY)
    assert app.applied_on == TODAY and app.next_follow_up == TODAY + timedelta(days=7)

    app = repo.change_status(ALICE, app.id, Status.INTERVIEW, note="Round 1", today=TODAY + timedelta(days=5))
    assert app.next_follow_up == TODAY + timedelta(days=8)

    app = repo.change_status(ALICE, app.id, Status.OFFER, today=TODAY + timedelta(days=9))
    assert app.next_follow_up is None
    assert [h.to_status for h in app.history] == [Status.SAVED, Status.APPLIED, Status.INTERVIEW, Status.OFFER]
    assert app.history[2].note == "Round 1"


def test_same_status_is_a_no_op(repo):
    app = new(repo, status=Status.APPLIED)
    assert len(repo.change_status(ALICE, app.id, Status.APPLIED).history) == 1


def test_withdrawing_a_saved_job_does_not_count_as_applied(repo):
    app = repo.change_status(ALICE, new(repo).id, Status.WITHDRAWN, today=TODAY)
    assert app.applied_on is None


# ---- reminders --------------------------------------------------------------------------
def test_due_reminders_order_and_exclusions(repo):
    late = new(repo, company="Late", status=Status.APPLIED, applied_on=TODAY - timedelta(days=20))
    soon = new(repo, company="Soon", status=Status.APPLIED, applied_on=TODAY - timedelta(days=7))
    new(repo, company="Future", status=Status.APPLIED)                     # due in 7 days
    new(repo, company="Closed", status=Status.REJECTED)
    new(repo, company="Saved")
    due = repo.due_reminders(ALICE, TODAY)
    assert [a.id for a in due] == [late.id, soon.id]                       # most overdue first
    assert repo.due_reminders(ALICE, TODAY + timedelta(days=7))[-1].company == "Future"


def test_mark_followed_up_snoozes_and_counts(repo):
    app = new(repo, status=Status.APPLIED, applied_on=TODAY - timedelta(days=10))
    app = repo.mark_followed_up(ALICE, app.id, today=TODAY)
    assert app.follow_up_count == 1 and app.next_follow_up == TODAY + timedelta(days=7)
    assert app.history[-1].note == "Followed up (#1)"
    assert repo.due_reminders(ALICE, TODAY) == []


def test_cannot_follow_up_on_closed_application(repo):
    app = new(repo, status=Status.GHOSTED)
    with pytest.raises(TrackerError):
        repo.mark_followed_up(ALICE, app.id)


# ---- editing ---------------------------------------------------------------------------
def test_update_fields_validates_and_whitelists(repo):
    app = new(repo, status=Status.APPLIED)
    app = repo.update_fields(ALICE, app.id, {"notes": "Spoke to HR", "contact_email": "hr@acme.com",
                                             "next_follow_up": TODAY + timedelta(days=2)})
    assert app.notes == "Spoke to HR" and app.next_follow_up == TODAY + timedelta(days=2)
    with pytest.raises(ValidationError):
        repo.update_fields(ALICE, app.id, {"contact_email": "nope"})
    with pytest.raises(TrackerError):
        repo.update_fields(ALICE, app.id, {"status": "offer"})            # status has its own method
    assert repo.get(ALICE, app.id).contact_email == "hr@acme.com"          # failed edit changed nothing


# ---- delete / limits -----------------------------------------------------------------------
def test_delete_and_delete_all_remove_history_too(repo):
    a, b = new(repo), new(repo, company="B")
    repo.delete(ALICE, a.id)
    assert [x.company for x in repo.list_applications(ALICE)] == ["B"]
    assert repo.delete_all(ALICE) == 1 and repo.list_applications(ALICE) == []
    with repo._sessions() as s:
        assert s.query(tracker_db.StatusChange).count() == 0


def test_per_user_limit(repo, monkeypatch):
    monkeypatch.setattr(tracker_db, "MAX_APPLICATIONS_PER_USER", 2)
    new(repo), new(repo)
    with pytest.raises(LimitError):
        new(repo)
    new(repo, user=BOB)  # other users are unaffected


# ---- stats through the repo --------------------------------------------------------------------
def test_stats_from_repo(repo):
    a = new(repo, status=Status.APPLIED, resume_version="v1")
    new(repo, status=Status.APPLIED, resume_version="v2")
    repo.change_status(ALICE, a.id, Status.INTERVIEW, today=TODAY)
    stats = repo.stats(ALICE, TODAY)
    assert stats["applied"] == 2 and stats["response_rate"] == 0.5
    assert stats["by_resume_version"]["v1"]["response_rate"] == 1.0


# ---- export / import -------------------------------------------------------------------------------
def test_export_import_round_trip_for_another_user(repo):
    a = new(repo, status=Status.APPLIED, notes="n", resume_version="v1", jd_text="JD")
    repo.change_status(ALICE, a.id, Status.INTERVIEW, note="R1", today=TODAY)
    text = repo.export_json(ALICE, TODAY)
    assert json.loads(text)["version"] == 1

    assert repo.import_json(BOB, text) == 1
    copy = repo.list_applications(BOB)[0]
    assert (copy.company, copy.status, copy.notes, copy.jd_text) == ("Acme", Status.INTERVIEW, "n", "JD")
    assert [h.to_status for h in copy.history] == [Status.APPLIED, Status.INTERVIEW]
    assert repo.get(ALICE, a.id).id != copy.id and len(repo.list_applications(ALICE)) == 1


def test_import_rejects_bad_files_without_partial_writes(repo):
    for bad in ("not json", '{"version": 1}',
                '{"version":1,"exported_on":"2026-10-18","applications":[{"company":"","role":"x"}]}'):
        with pytest.raises(ValueError):
            repo.import_json(ALICE, bad)
    assert repo.list_applications(ALICE) == []


def test_import_respects_limit(repo, monkeypatch):
    new(repo), new(repo)
    text = repo.export_json(ALICE, TODAY)
    monkeypatch.setattr(tracker_db, "MAX_APPLICATIONS_PER_USER", 3)
    with pytest.raises(LimitError):
        repo.import_json(ALICE, text)  # 2 existing + 2 imported > 3
    assert len(repo.list_applications(ALICE)) == 2
