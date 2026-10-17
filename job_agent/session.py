"""Session handling: every user gets an isolated folder for their index.

The session id is random, and we validate it strictly so nobody can use
a crafted id like '../../etc' to escape the data folder (path traversal).
"""

import re
import shutil
import uuid
from pathlib import Path

from job_agent import config

_SESSION_RE = re.compile(r"^[a-f0-9]{32}$")


def new_session_id() -> str:
    return uuid.uuid4().hex


def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not _SESSION_RE.fullmatch(session_id):
        raise ValueError("Invalid session id.")
    return session_id


def session_dir(session_id: str, create: bool = False) -> Path:
    path = config.DATA_DIR / validate_session_id(session_id)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def delete_session(session_id: str) -> None:
    """Remove everything stored for a session (privacy: 'delete my data')."""
    path = session_dir(session_id)
    if path.exists():
        shutil.rmtree(path)
