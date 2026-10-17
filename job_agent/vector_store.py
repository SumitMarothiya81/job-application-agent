"""A tiny exact-search vector store (numpy + cosine similarity).

Why not Chroma/FAISS yet? One resume is ~30-80 chunks. Exact search over
that is instant and has zero dependencies, so a vector database would add
install weight and failure modes (e.g. SQLite issues on free hosts) for no
gain. Everything goes through this class, so swapping in Chroma or FAISS
later is a one-file change.
"""

import json
from pathlib import Path

import numpy as np

from job_agent.resume_schema import EvidenceChunk


class IndexMismatchError(RuntimeError):
    """The index was built with a different embedding model: re-index needed."""


class VectorStore:
    def __init__(self, chunks: list[EvidenceChunk], vectors: np.ndarray, embedding_id: str):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.embedding_id = embedding_id

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[tuple[EvidenceChunk, float]]:
        """Top-k chunks by cosine similarity (vectors are pre-normalized)."""
        if len(self.chunks) == 0:
            return []
        scores = self.vectors @ query_vector
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors, allow_pickle=False)
        payload = {
            "embedding_id": self.embedding_id,
            "chunks": [c.model_dump() for c in self.chunks],
        }
        (folder / "chunks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, expected_embedding_id: str) -> "VectorStore":
        payload = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        if payload["embedding_id"] != expected_embedding_id:
            raise IndexMismatchError(
                f"Index built with '{payload['embedding_id']}' but current setup is "
                f"'{expected_embedding_id}'. Re-index the resume."
            )
        vectors = np.load(folder / "vectors.npy", allow_pickle=False)
        chunks = [EvidenceChunk(**c) for c in payload["chunks"]]
        return cls(chunks, vectors, payload["embedding_id"])
