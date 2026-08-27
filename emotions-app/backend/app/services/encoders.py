"""Thin protocols over the ML primitives the scorer needs.

Scoring code depends only on these interfaces, never on sentence-transformers
directly, so tests can substitute deterministic fakes (see tests/conftest.py).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class TextEncoder(Protocol):
    """Sentence/paragraph embedder. `SentenceTransformer` satisfies this as-is."""

    def encode(
        self, sentences: list[str], *, normalize_embeddings: bool = ..., **kwargs: object
    ) -> np.ndarray: ...


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity for L2-normalised or unnormalised inputs."""
    a = np.atleast_2d(a).astype(np.float32)
    b = np.atleast_2d(b).astype(np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T
