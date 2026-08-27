"""Metrics that need only spaCy, not embeddings - clarity, specificity, confidence, conciseness."""

from __future__ import annotations

import pytest

from app.services.scoring.clarity import score_clarity
from app.services.scoring.conciseness import score_conciseness
from app.services.scoring.confidence import score_confidence
from app.services.scoring.specificity import score_specificity
from app.services.scoring.text_features import extract_features

STRONG = (
    "I led the migration of our billing pipeline to Airflow over six weeks. "
    "I profiled the existing jobs, found that 40 percent of runtime was one unindexed join, "
    "and rewrote it. The nightly run dropped from four hours to 35 minutes."
)
HEDGY = (
    "I think I maybe helped with the pipeline a bit. I'm not really sure how much I did, "
    "but I guess it sort of worked out okay in the end, probably."
)
RAMBLING = (
    "So the thing about the pipeline is that pipelines are really important and the pipeline "
    "was a pipeline that needed work and so we worked on the pipeline because the pipeline "
    "mattered and pipelines matter a lot and the pipeline is now a better pipeline."
)


@pytest.fixture
def feats(nlp):
    def _make(text: str):
        return extract_features(text, nlp)

    return _make


def test_specificity_rewards_concrete_detail(feats):
    strong = score_specificity(feats(STRONG))
    vague = score_specificity(feats(HEDGY))
    assert strong.value > vague.value + 15


def test_confidence_penalises_hedging(feats):
    assertive = score_confidence(feats(STRONG))
    hedgy = score_confidence(feats(HEDGY))
    assert assertive.value > hedgy.value + 20
    assert hedgy.evidence["hedges_per_sentence"] > 1


def test_conciseness_penalises_repetition(feats):
    tight = score_conciseness(feats(STRONG))
    padded = score_conciseness(feats(RAMBLING))
    assert tight.value > padded.value + 15


def test_clarity_penalises_one_giant_sentence(feats):
    wall = "This is a single enormous run on sentence that just keeps going " * 8
    choppy_ok = score_clarity(feats(STRONG))
    assert choppy_ok.value > score_clarity(feats(wall)).value


def test_confidence_blends_prosody_when_present(feats):
    text_only = score_confidence(feats(STRONG))
    with_bad_voice = score_confidence(feats(STRONG), prosody_steadiness=0.1)
    assert with_bad_voice.value < text_only.value
    assert "prosody_steadiness" in with_bad_voice.evidence
