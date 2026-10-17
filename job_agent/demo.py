"""Demo mode: a ready-made profile of the (fictional) sample resume.

The profile is stored as JSON so the public demo needs no resume parsing and
no Gemini call for the resume; only a quick local embedding of ~11 chunks.
"""

from pathlib import Path

from job_agent import embeddings
from job_agent.chunking import embedding_text, profile_to_chunks
from job_agent.resume_schema import ResumeProfile
from job_agent.vector_store import VectorStore

DEMO_PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "demo" / "profile.json"


def load_demo_profile() -> ResumeProfile:
    return ResumeProfile.model_validate_json(DEMO_PROFILE_PATH.read_text(encoding="utf-8"))


def build_store(profile: ResumeProfile) -> VectorStore:
    """In-memory vector store for a profile (nothing is written to disk)."""
    chunks = profile_to_chunks(profile)
    vectors = embeddings.embed_passages([embedding_text(c) for c in chunks])
    return VectorStore(chunks, vectors, embeddings.embedding_id())
