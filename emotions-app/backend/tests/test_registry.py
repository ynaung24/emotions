"""Startup invariants: build each model once, and stay within the encoder budget."""

from __future__ import annotations

import pytest

from app.services.evaluator import Evaluator
from app.services.feedback.judge import NullJudge

pytestmark = pytest.mark.slow


def test_each_model_built_exactly_once(real_registry):
    # warmup already ran in the fixture; touching the properties again must not rebuild
    _ = real_registry.encoder
    _ = real_registry.nlp
    assert real_registry.build_counts.get("encoder") == 1
    assert real_registry.build_counts.get("nlp") == 1
    for name, count in real_registry.build_counts.items():
        assert count == 1, f"{name} built {count} times"


def test_text_evaluation_stays_within_two_encoder_passes(settings, real_registry, monkeypatch):
    calls = {"n": 0}
    real_encode = real_registry.encoder.encode

    def counting_encode(*args, **kwargs):
        calls["n"] += 1
        return real_encode(*args, **kwargs)

    monkeypatch.setattr(real_registry.encoder, "encode", counting_encode)

    evaluator = Evaluator(settings, real_registry, NullJudge())
    evaluator.evaluate_text(
        "Tell me about your background in data science.",
        "I have a statistics degree and four years building models in Python with scikit-learn.",
    )
    assert calls["n"] <= 2
