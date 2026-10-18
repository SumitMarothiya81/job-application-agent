"""Database layer for the application tracker (SQLAlchemy 2.0).

Default is a local SQLite file. Set DATABASE_URL to a Postgres URL later and the
same code runs unchanged. EVERY method takes a user_id and filters on it, so one
visitor can never read or change another visitor's applications.
"""

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    selectinload,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool

from job_agent.session import validate_session_id
from job_agent.tracker_logic import (
    DEFAULT_SNOOZE_DAYS,
    FOLLOW_UP_AFTER_DAYS,
    MAX_APPLICATIONS_PER_USER,
    NOT_APPLIED,
    TERMINAL,
    ApplicationInput,
    ApplicationView,
    Status,
    TrackerExport,
    compute_stats,
    schedule_follow_up,
    today_local,
)


class TrackerError(ValueError):
    """Base class for tracker problems that are safe to show to the user."""


class NotFoundError(TrackerError):
    pass


class LimitError(TrackerError):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)  # stored as naive UTC


class Base(DeclarativeBase):
    pass


class Application(Base):
    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    company: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20), default=Status.SAVED.value)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    location: Mapped[str | None] = mapped_column(String(120), nullable=True)
    match_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resume_version: Mapped[str | None] = mapped_column(String(60), nullable=True)
    contact_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    contact_email: Mapped[str | None] = mapped_column(String(200), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    jd_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    applied_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    next_follow_up: Mapped[date | None] = mapped_column(Date, nullable=True)
    follow_up_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    history: Mapped[list["StatusChange"]] = relationship(
        back_populates="application", cascade="all, delete-orphan", order_by="StatusChange.id"
    )


class StatusChange(Base):
    __tablename__ = "status_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[int] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"))
    from_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    to_status: Mapped[str] = mapped_column(String(20))
    changed_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    note: Mapped[str | None] = mapped_column(String(500), nullable=True)

    application: Mapped[Application] = relationship(back_populates="history")


def get_engine(url: str | None = None):
    url = url or os.getenv("DATABASE_URL", "sqlite:///data/tracker.db")
    if url.startswith("postgres://"):  # some hosts still hand out the old scheme
        url = url.replace("postgres://", "postgresql://", 1)
    kwargs: dict = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}  # Streamlit uses threads
        if ":memory:" in url or url == "sqlite://":
            kwargs["poolclass"] = StaticPool
        else:
            Path(url.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, **kwargs)


_EDITABLE = ("source_url", "location", "match_score", "resume_version", "contact_name",
             "contact_email", "notes", "next_follow_up")


class TrackerRepo:
    def __init__(self, engine):
        Base.metadata.create_all(engine)
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _owned(session, user_id: str, app_id: int) -> Application:
        row = session.scalar(
            select(Application)
            .options(selectinload(Application.history))
            .where(Application.id == app_id, Application.user_id == user_id)
        )
        if row is None:
            raise NotFoundError("Application not found.")
        return row

    @staticmethod
    def _count(session, user_id: str) -> int:
        return session.scalar(
            select(func.count()).select_from(Application).where(Application.user_id == user_id)
        )

    # ---- create -----------------------------------------------------------
    def add(self, user_id: str, data: ApplicationInput, today: date | None = None) -> ApplicationView:
        validate_session_id(user_id)
        today = today or today_local()
        with self._sessions.begin() as session:
            if self._count(session, user_id) >= MAX_APPLICATIONS_PER_USER:
                raise LimitError(f"Limit of {MAX_APPLICATIONS_PER_USER} applications reached.")

            applied_on = data.applied_on
            if applied_on is None and data.status not in NOT_APPLIED:
                applied_on = today
            follow_up = data.next_follow_up
            if follow_up is None and data.status in FOLLOW_UP_AFTER_DAYS:
                anchor = applied_on if data.status == Status.APPLIED else today
                follow_up = schedule_follow_up(data.status, anchor)
            if data.status in TERMINAL:
                follow_up = None

            fields = data.model_dump(exclude={"status", "applied_on", "next_follow_up"})
            row = Application(user_id=user_id, status=data.status.value, applied_on=applied_on,
                              next_follow_up=follow_up, **fields)
            row.history.append(StatusChange(from_status=None, to_status=data.status.value, note="Created"))
            session.add(row)
            session.flush()
            return ApplicationView.model_validate(row)

    # ---- read -------------------------------------------------------------
    def get(self, user_id: str, app_id: int) -> ApplicationView:
        validate_session_id(user_id)
        with self._sessions() as session:
            return ApplicationView.model_validate(self._owned(session, user_id, app_id))

    def list_applications(self, user_id: str, status: Status | None = None) -> list[ApplicationView]:
        validate_session_id(user_id)
        query = (select(Application).options(selectinload(Application.history))
                 .where(Application.user_id == user_id).order_by(Application.id.desc()))
        if status is not None:
            query = query.where(Application.status == status.value)
        with self._sessions() as session:
            return [ApplicationView.model_validate(r) for r in session.scalars(query)]

    def due_reminders(self, user_id: str, today: date | None = None) -> list[ApplicationView]:
        """Open applications whose follow-up date is today or already past, oldest first."""
        today = today or today_local()
        due = [a for a in self.list_applications(user_id)
               if a.status not in TERMINAL and a.next_follow_up is not None and a.next_follow_up <= today]
        return sorted(due, key=lambda a: (a.next_follow_up, a.id))

    def stats(self, user_id: str, today: date | None = None) -> dict:
        today = today or today_local()
        return compute_stats(self.list_applications(user_id), today)

    # ---- update -----------------------------------------------------------
    def update_fields(self, user_id: str, app_id: int, updates: dict) -> ApplicationView:
        """Edit details (not status). Values are validated by the same rules as new entries."""
        validate_session_id(user_id)
        unknown = set(updates) - set(_EDITABLE)
        if unknown:
            raise TrackerError(f"Cannot edit: {', '.join(sorted(unknown))}")
        with self._sessions.begin() as session:
            row = self._owned(session, user_id, app_id)
            current = ApplicationInput.model_validate(ApplicationView.model_validate(row).model_dump())
            merged = ApplicationInput.model_validate({**current.model_dump(), **updates})
            for name in _EDITABLE:
                setattr(row, name, getattr(merged, name))
            return ApplicationView.model_validate(row)

    def change_status(self, user_id: str, app_id: int, new_status: Status, note: str | None = None,
                      today: date | None = None) -> ApplicationView:
        validate_session_id(user_id)
        today = today or today_local()
        with self._sessions.begin() as session:
            row = self._owned(session, user_id, app_id)
            old = Status(row.status)
            if old == new_status:
                return ApplicationView.model_validate(row)

            row.status = new_status.value
            if new_status not in NOT_APPLIED and row.applied_on is None:
                row.applied_on = today
            if new_status in TERMINAL or new_status not in FOLLOW_UP_AFTER_DAYS:
                row.next_follow_up = None
            else:
                anchor = row.applied_on if new_status == Status.APPLIED else today
                row.next_follow_up = schedule_follow_up(new_status, anchor)
            row.history.append(StatusChange(from_status=old.value, to_status=new_status.value,
                                            note=(note or "").strip()[:500] or None))
            session.flush()
            return ApplicationView.model_validate(row)

    def mark_followed_up(self, user_id: str, app_id: int, today: date | None = None,
                         snooze_days: int = DEFAULT_SNOOZE_DAYS) -> ApplicationView:
        validate_session_id(user_id)
        today = today or today_local()
        with self._sessions.begin() as session:
            row = self._owned(session, user_id, app_id)
            status = Status(row.status)
            if status in TERMINAL:
                raise TrackerError("This application is closed; no follow-up needed.")
            row.follow_up_count += 1
            row.next_follow_up = today + timedelta(days=snooze_days)
            row.history.append(StatusChange(from_status=status.value, to_status=status.value,
                                            note=f"Followed up (#{row.follow_up_count})"))
            session.flush()
            return ApplicationView.model_validate(row)

    # ---- delete -----------------------------------------------------------
    def delete(self, user_id: str, app_id: int) -> None:
        validate_session_id(user_id)
        with self._sessions.begin() as session:
            session.delete(self._owned(session, user_id, app_id))

    def delete_all(self, user_id: str) -> int:
        validate_session_id(user_id)
        with self._sessions.begin() as session:
            rows = list(session.scalars(select(Application).where(Application.user_id == user_id)))
            for row in rows:
                session.delete(row)  # ORM delete, so the history rows go too
            return len(rows)

    # ---- export / import (users keep their own copy of their data) ------------
    def export_json(self, user_id: str, today: date | None = None) -> str:
        today = today or today_local()
        payload = TrackerExport.model_validate(
            {"exported_on": today, "applications": [a.model_dump() for a in self.list_applications(user_id)]}
        )
        return payload.model_dump_json(indent=2)

    def import_json(self, user_id: str, text: str) -> int:
        """Add the applications from an export file. Raises on invalid or oversized files."""
        validate_session_id(user_id)
        payload = TrackerExport.model_validate_json(text)
        with self._sessions.begin() as session:
            if self._count(session, user_id) + len(payload.applications) > MAX_APPLICATIONS_PER_USER:
                raise LimitError(f"Import would exceed the limit of {MAX_APPLICATIONS_PER_USER} applications.")
            for item in payload.applications:
                fields = item.model_dump(exclude={"status", "history", "follow_up_count"})
                row = Application(user_id=user_id, status=item.status.value,
                                  follow_up_count=item.follow_up_count, **fields)
                for change in item.history:
                    row.history.append(StatusChange(
                        from_status=change.from_status.value if change.from_status else None,
                        to_status=change.to_status.value, changed_at=change.changed_at, note=change.note))
                if not row.history:
                    row.history.append(StatusChange(from_status=None, to_status=item.status.value, note="Imported"))
                session.add(row)
            return len(payload.applications)
