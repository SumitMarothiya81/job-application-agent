"""Local embeddings behind two small functions, so the backend is swappable.

Backends (env EMBEDDING_BACKEND):
  fastembed (default) - ONNX models running locally: free, no rate limits,
                        no PyTorch, small enough for free hosting tiers.
  fake                - tiny hashing embedder used ONLY in unit tests.

IMPORTANT: vectors from different models are not comparable. The index
remembers which embedding it was built with (embedding_id) and refuses to
be queried with a different one.
"""

import hashlib
import os
import re
from functools import lru_cache

import numpy as np

from job_agent.config import DEFAULT_EMBEDDING_MODEL


def _backend() -> str:
    return os.getenv("EMBEDDING_BACKEND", "fastembed").lower()


def _model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embedding_id() -> str:
    """Identifies the embedding setup; stored with every index."""
    return f"{_backend()}:{_model_name()}"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# ---- fake backend (tests) -------------------------------------------------
_FAKE_DIM = 256
_TOKEN = re.compile(r"[a-z0-9\-\+\.#]+")


def _fake_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), _FAKE_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in _TOKEN.findall(text.lower()):
            index = int(hashlib.md5(token.encode()).hexdigest(), 16) % _FAKE_DIM
            out[row, index] += 1.0
    return out


# ---- fastembed backend ----------------------------------------------------
@lru_cache(maxsize=1)
def _fastembed_model(name: str):
    from fastembed import TextEmbedding  # imported lazily: heavy import

    return TextEmbedding(model_name=name)  # downloads once, then cached on disk


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed documents (resume chunks). Returns an (n, dim) float32 array."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    if _backend() == "fake":
        return _normalize(_fake_embed(texts))
    model = _fastembed_model(_model_name())
    vectors = np.array(list(model.passage_embed(texts)), dtype=np.float32)
    return _normalize(vectors)


def embed_query(text: str) -> np.ndarray:
    """Embed a search query. Returns a (dim,) float32 vector.

    BGE models are trained so queries and passages are embedded slightly
    differently; fastembed's query_embed handles that for us.
    """
    if _backend() == "fake":
        return _normalize(_fake_embed([text]))[0]
    model = _fastembed_model(_model_name())
    vector = np.array(list(model.query_embed([text])), dtype=np.float32)
    return _normalize(vector)[0]
