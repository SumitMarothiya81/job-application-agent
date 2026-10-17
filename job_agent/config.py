"""Central place for configuration. Everything else imports from here."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Comma-separated list, tried in order. If the first model is overloaded (503)
# or rate-limited (429), we automatically fall back to the next one.
# GEMINI_MODEL (single name) still works for backward compatibility.
_raw_models = os.getenv("GEMINI_MODELS") or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash"
GEMINI_MODELS = [m.strip().removeprefix("models/") for m in _raw_models.split(",") if m.strip()]
GEMINI_MODEL = GEMINI_MODELS[0]  # primary model

# Embeddings (read at call time in embeddings.py so tests can switch backend)
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Where per-session indexes are stored (git-ignored)
DATA_DIR = Path(os.getenv("DATA_DIR", "data/sessions"))

# Upload safety
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))
ALLOWED_RESUME_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
def require_api_key() -> str:
    """Fail early with a clear message instead of a confusing API error."""
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key."
        )
    return GOOGLE_API_KEY
