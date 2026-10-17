"""Build and query a per-session resume index (the RAG part).

Pipeline:  file -> text -> redact PII -> Gemini structured parse
           -> one chunk per bullet -> local embeddings -> saved in session folder
"""

import json
from pathlib import Path

from job_agent import embeddings
from job_agent.chunking import embedding_text, profile_to_chunks
from job_agent.privacy import redact_pii
from job_agent.resume_loader import load_resume_text
from job_agent.resume_parser import parse_resume
from job_agent.resume_schema import EvidenceChunk, ResumeProfile
from job_agent.session import session_dir
from job_agent.vector_store import VectorStore


def index_profile(profile: ResumeProfile, session_id: str) -> VectorStore:
    """Chunk + embed + save an already-parsed profile."""
    chunks = profile_to_chunks(profile)
    if not chunks:
        raise ValueError("No usable content found in the resume (no bullets/skills).")
    vectors = embeddings.embed_passages([embedding_text(c) for c in chunks])
    store = VectorStore(chunks, vectors, embeddings.embedding_id())

    folder = session_dir(session_id, create=True)
    store.save(folder)
    (folder / "profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return store


def build_resume_index(data: bytes, filename: str, session_id: str) -> ResumeProfile:
    """Full pipeline from an uploaded file. Returns the parsed profile."""
    raw_text = load_resume_text(data, filename)
    safe_text = redact_pii(raw_text)  # only this version ever reaches the LLM
    profile = parse_resume(safe_text)
    index_profile(profile, session_id)
    return profile


def load_profile(session_id: str) -> ResumeProfile:
    path = session_dir(session_id) / "profile.json"
    return ResumeProfile.model_validate_json(path.read_text(encoding="utf-8"))


def load_store(session_id: str) -> VectorStore:
    return VectorStore.load(session_dir(session_id), embeddings.embedding_id())


def retrieve_evidence(
    session_id: str, query: str, k: int = 3, store: VectorStore | None = None
) -> list[tuple[EvidenceChunk, float]]:
    """Find the resume chunks most relevant to a query (skill, requirement...)."""
    store = store or load_store(session_id)
    return store.search(embeddings.embed_query(query), k=k)
