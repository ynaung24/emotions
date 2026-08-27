"""Relevance / completeness / structure with the fake encoder - logic, not model quality."""

from __future__ import annotations

import numpy as np

from app.services.scoring.completeness import score_completeness
from app.services.scoring.relevance import score_relevance
from app.services.scoring.structure import score_structure
from tests.conftest import _hash_vec


def _vecs(texts: list[str]) -> np.ndarray:
    return np.vstack([_hash_vec(t) for t in texts])


def test_relevance_higher_for_on_topic():
    question = "how do you deploy a machine learning model to production"
    on_topic = "I containerize the model and deploy it to production with a canary rollout"
    off_topic = "My favourite hobby is baking sourdough bread on weekends"

    qv = _vecs([question])
    on = score_relevance(question, on_topic, qv, _vecs([on_topic]), _vecs([on_topic]))
    off = score_relevance(question, off_topic, qv, _vecs([off_topic]), _vecs([off_topic]))
    assert on.value > off.value


def test_completeness_lexical_match_covers_keywords():
    keywords = ("docker", "kubernetes", "monitoring", "rollback")
    sentence = "I put the model in a docker container and add monitoring"
    vocab = frozenset(sentence.lower().split())
    out = score_completeness(keywords, _vecs(list(keywords)), _vecs([sentence]), vocab)
    assert set(out.evidence["covered"]) == {"docker", "monitoring"}
    assert set(out.evidence["missing"]) == {"kubernetes", "rollback"}


def test_completeness_not_computed_without_keywords():
    out = score_completeness((), np.zeros((0, 1)), _vecs(["anything"]), frozenset())
    assert out.computed is False


def test_structure_detects_star_components():
    star_protos = _vecs(
        [
            "the situation and background context was",
            "my task and responsibility was",
            "the actions and steps I took were",
            "the result outcome and impact was",
        ]
    )
    full = _vecs(
        [
            "the situation and background context was a failing pipeline",
            "my task and responsibility was to restore it",
            "the actions and steps I took were adding checks",
            "the result outcome and impact was eight months of clean runs",
        ]
    )
    out = score_structure(full, star_protos)
    assert out.value > 60
    assert set(out.evidence["components_present"]) == {"situation", "task", "action", "result"}
