"""Shared fixtures.

`fake_encoder` / `fake_pair_scorer` keep the pure-scoring unit tests fast and
deterministic. `real_registry` (session-scoped, `slow`) loads the actual models
for the golden discrimination tests and the API integration test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings, get_settings
from app.services.corpus import load_corpus

_DIM = 64
REPO_ROOT = Path(__file__).resolve().parents[3]


def _hash_vec(text: str) -> np.ndarray:
    """Deterministic pseudo-embedding: hash bag-of-words into a fixed vector so
    that texts sharing words land near each other in cosine space."""
    vec = np.zeros(_DIM, dtype=np.float32)
    for tok in text.lower().split():
        h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
        vec[h % _DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, sentences: list[str], *, normalize_embeddings: bool = True, **_: object):
        self.calls += 1
        return np.vstack([_hash_vec(s) for s in sentences])


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture(scope="session")
def settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture(scope="session")
def corpus(settings: Settings):
    return load_corpus(settings.corpus_path)


@pytest.fixture(scope="session")
def nlp():
    import spacy

    return spacy.load("en_core_web_md", disable=["textcat"])


@pytest.fixture(scope="session")
def real_registry(settings: Settings, corpus):
    pytest.importorskip("sentence_transformers")
    from app.services.registry import ModelRegistry

    reg = ModelRegistry(settings, corpus)
    reg.warmup()
    return reg


@pytest.fixture(scope="session")
def golden_cases() -> list[dict[str, str]]:
    import json

    path = Path(__file__).parent / "data" / "golden.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
