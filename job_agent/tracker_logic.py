"""Application-tracking rules: statuses, follow-up scheduling, stats, calendar export.

Pure Python (no database, no Streamlit) so every rule is easy to test.
"""

import os
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Status(str, Enum):
    SAVED = "saved"
    APPLIED = "applied"
    ONLINE_ASSESSMENT = "online_assessment"
    INTERVIEW = "interview"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    GHOSTED = "ghosted"


STATUS_LABELS = {
    Status.SAVED: "Saved",
    Status.APPLIED: "Applied",
    Status.ONLINE_ASSESSMENT: "Online assessment",
    Status.INTERVIEW: "Interview",
    Status.OFFER: "Offer",
    Status.REJECTED: "Rejected",
    Status.WITHDRAWN: "Withdrawn",
    Status.GHOSTED: "No response (ghosted)",
}

TERMINAL = {Status.OFFER, Status.REJECTED, Status.WITHDRAWN, Status.GHOSTED}
NOT_APPLIED = {Status.SAVED, Status.WITHDRAWN}  # these never set an 'applied on' date
RESPONSE_STATUSES = {Status.ONLINE_ASSESSMENT, Status.INTERVIEW, Status.OFFER, Status.REJECTED}
INTERVIEW_STATUSES = {Status.INTERVIEW, Status.OFFER}

# Days until the next follow-up reminder after entering a status.
FOLLOW_UP_AFTER_DAYS = {Status.APPLIED: 7, Status.ONLINE_ASSESSMENT: 3, Status.INTERVIEW: 3}
DEFAULT_SNOOZE_DAYS = 7
MAX_APPLICATIONS_PER_USER = 300


def today_local(tz_name: str | None = None) -> date:
    """Today's date in the user's timezone (hosting servers run on UTC)."""
    name = tz_name or os.getenv("TRACKER_TIMEZONE", "Asia/Kolkata")
    try:
        return datetime.now(ZoneInfo(name)).date()
    except Exception:  # noqa: BLE001  (unknown zone or no tz database)
        return datetime.now(timezone.utc).date()


def to_local(moment: datetime, tz_name: str | None = None) -> datetime:
    """Convert a stored (naive UTC) timestamp to the user's timezone for display."""
    name = tz_name or os.getenv("TRACKER_TIMEZONE", "Asia/Kolkata")
    try:
        return moment.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(name))
    except Exception:  # noqa: BLE001
        return moment


def schedule_follow_up(status: Status, anchor: date) -> date | None:
    """Reminder date for a status, counted from `anchor` (None if no reminder applies)."""
    days = FOLLOW_UP_AFTER_DAYS.get(status)
    return anchor + timedelta(days=days) if days else None


# --------------------------------------------------------------------------
# Data models (validated input, and read-only views)
# --------------------------------------------------------------------------
_OPTIONAL_TEXT = ("source_url", "location", "resume_version", "contact_name",
                  "contact_email", "notes", "jd_text")


class ApplicationInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    company: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=120)
    status: Status = Status.SAVED
    source_url: str | None = Field(default=None, max_length=500)
    location: str | None = Field(default=None, max_length=120)
    match_score: int | None = Field(default=None, ge=0, le=100)
    resume_version: str | None = Field(default=None, max_length=60)
    contact_name: str | None = Field(default=None, max_length=120)
    contact_email: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)
    jd_text: str | None = Field(default=None, max_length=20000)
    applied_on: date | None = None
    next_follow_up: date | None = None

    @field_validator(*_OPTIONAL_TEXT, mode="before")
    @classmethod
    def _blank_to_none(cls, value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("source_url")
    @classmethod
    def _check_url(cls, value):
        if value and not re.match(r"^https?://\S+$", value, re.IGNORECASE):
            raise ValueError("must start with http:// or https:// and contain no spaces")
        return value

    @field_validator("contact_email")
    @classmethod
    def _check_email(cls, value):
        if value and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value):
            raise ValueError("is not a valid email address")
        return value


class StatusChangeView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    from_status: Status | None = None
    to_status: Status
    changed_at: datetime
    note: str | None = None


class ApplicationView(ApplicationInput):
    """What the database returns: the input fields plus id, history and counters."""

    model_config = ConfigDict(from_attributes=True, str_strip_whitespace=True)

    id: int
    follow_up_count: int = 0
    created_at: datetime
    history: list[StatusChangeView] = Field(default_factory=list)


class ExportedChange(BaseModel):
    from_status: Status | None = None
    to_status: Status
    changed_at: datetime
    note: str | None = Field(default=None, max_length=500)


class ExportedApplication(ApplicationInput):
    follow_up_count: int = Field(default=0, ge=0, le=100)
    history: list[ExportedChange] = Field(default_factory=list, max_length=100)


class TrackerExport(BaseModel):
    """File format for 'Export my tracker' / 'Import'. Users keep their own data."""

    version: Literal[1] = 1
    exported_on: date
    applications: list[ExportedApplication] = Field(max_length=MAX_APPLICATIONS_PER_USER)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def _ever(app: ApplicationView, statuses: set[Status]) -> bool:
    return app.status in statuses or any(h.to_status in statuses for h in app.history)


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def compute_stats(apps: list[ApplicationView], today: date) -> dict:
    """Pipeline numbers. 'Applied' = has an applied date; 'responded' = ever reached an
    assessment, interview, offer or rejection (a rejection is still a reply)."""
    applied = [a for a in apps if a.applied_on is not None]
    responded = [a for a in applied if _ever(a, RESPONSE_STATUSES)]
    interviewed = [a for a in applied if _ever(a, INTERVIEW_STATUSES)]

    by_version: dict[str, dict] = {}
    for app in applied:
        label = app.resume_version or "(no label)"
        entry = by_version.setdefault(label, {"applied": 0, "responded": 0})
        entry["applied"] += 1
        entry["responded"] += int(_ever(app, RESPONSE_STATUSES))
    for entry in by_version.values():
        entry["response_rate"] = _rate(entry["responded"], entry["applied"])

    monday = today - timedelta(days=today.weekday())
    week_starts = [monday - timedelta(weeks=i) for i in range(7, -1, -1)]
    weekly = [
        {"week_start": start,
         "applications": sum(1 for a in applied if start <= a.applied_on < start + timedelta(days=7))}
        for start in week_starts
    ]

    return {
        "total": len(apps),
        "by_status": dict(Counter(a.status for a in apps)),
        "applied": len(applied),
        "responded": len(responded),
        "interviewed": len(interviewed),
        "response_rate": _rate(len(responded), len(applied)),
        "interview_rate": _rate(len(interviewed), len(applied)),
        "by_resume_version": by_version,
        "weekly": weekly,
    }


# --------------------------------------------------------------------------
# Calendar export (.ics), so reminders show up in Google/Apple/Outlook calendars
# --------------------------------------------------------------------------
def _ics_escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
            .replace("\r\n", "\\n").replace("\n", "\\n"))


def _fold(line: str, width: int = 70) -> list[str]:
    """RFC 5545: long lines are split, continuation lines start with a space."""
    chunks = [line[i:i + width] for i in range(0, len(line), width)] or [""]
    return [chunks[0]] + [" " + c for c in chunks[1:]]


def follow_up_ics(app: ApplicationView, now: datetime | None = None) -> str:
    if app.next_follow_up is None:
        raise ValueError("This application has no follow-up date.")
    now = now or datetime.now(timezone.utc)
    start = app.next_follow_up
    description = f"Follow up on your application to {app.company} for {app.role}. " \
                  f"Status: {STATUS_LABELS[app.status]}."
    if app.contact_name or app.contact_email:
        description += f" Contact: {app.contact_name or ''} {app.contact_email or ''}".rstrip() + "."
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Job Application Agent//EN", "BEGIN:VEVENT",
        f"UID:application-{app.id}-{start:%Y%m%d}@job-application-agent",
        f"DTSTAMP:{now:%Y%m%dT%H%M%SZ}",
        f"DTSTART;VALUE=DATE:{start:%Y%m%d}",
        f"DTEND;VALUE=DATE:{start + timedelta(days=1):%Y%m%d}",
        f"SUMMARY:{_ics_escape(f'Follow up: {app.company} - {app.role}')}",
        f"DESCRIPTION:{_ics_escape(description)}",
        "END:VEVENT", "END:VCALENDAR",
    ]
    folded = [piece for line in lines for piece in _fold(line)]
    return "\r\n".join(folded) + "\r\n"
