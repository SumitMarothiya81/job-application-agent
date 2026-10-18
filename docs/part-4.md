# Part 4: Application tracker: what we built and why

(Rewrite in your own words. Questions to answer:)
- Why SQLAlchemy instead of raw sqlite3? (one URL change moves SQLite -> Postgres)
- Why does EVERY repository method take a user_id and filter by it? How do the tests prove isolation?
- Why keep a status_history table instead of only the current status? (response-rate stats, audit trail)
- Why is a rejection counted as a "reply" in the response rate?
- Why are the rules (follow-up dates, stats, .ics) in a pure module with no database or UI?
- Why in-app reminders + .ics export instead of email/push? (free hosting sleeps; no scheduler)
- Why export/import JSON, and why validate imports with Pydantic (size limits, enums)?
- Why a per-user cap on applications on a public demo?
- Why did the tracker tab have to be drawn after the match tab in app.py? (Streamlit runs top to bottom)
- Why name the method list_applications and not list? (it shadows the builtin inside the class)
