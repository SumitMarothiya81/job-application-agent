"""Usage caps that protect the server's Gemini quota in public demo mode.

Two limits: per visitor session, and per UTC day for the whole app. The daily
cap is the real protection (a visitor can refresh the page to get a new
session). Counts live in memory, so they reset when the app restarts.
"""

import threading
from datetime import datetime, timezone


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class UsageLimiter:
    def __init__(self, per_session: int, per_day: int, today=_utc_today):
        self.per_session = per_session
        self.per_day = per_day
        self._today = today
        self._lock = threading.Lock()
        self._day = today()
        self._day_used = 0
        self._sessions: dict[str, int] = {}

    def _roll_day(self) -> None:
        day = self._today()
        if day != self._day:
            self._day, self._day_used, self._sessions = day, 0, {}

    def try_consume(self, session_key: str, units: int = 1) -> tuple[bool, str]:
        """Reserve `units` calls. Returns (allowed, message_if_denied)."""
        with self._lock:
            self._roll_day()
            used = self._sessions.get(session_key, 0)
            if self._day_used + units > self.per_day:
                return False, "The demo's daily quota is used up. Add your own Gemini key to continue."
            if used + units > self.per_session:
                return False, "Demo limit reached for this session. Add your own Gemini key to continue."
            self._sessions[session_key] = used + units
            self._day_used += units
            return True, ""

    def remaining(self, session_key: str) -> int:
        with self._lock:
            self._roll_day()
            return max(0, min(self.per_session - self._sessions.get(session_key, 0),
                              self.per_day - self._day_used))
