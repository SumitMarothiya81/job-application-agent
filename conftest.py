"""Pytest setup: use a tiny fake embedder so tests never download models."""

import os

os.environ["EMBEDDING_BACKEND"] = "fake"