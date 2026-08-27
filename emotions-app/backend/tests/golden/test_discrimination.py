"""The regression guard for the whole defect class.

The OLD implementation scored every normal-length answer >= 65/100 and could not
separate a strong answer from an off-topic one. These tests assert real
separation. If they fail, the scorer has regressed to noise.

Tiers per question in golden.jsonl:
  strong   - detailed, specific, correct
  mediocre - on topic, thin, no specifics
  weak     - on topic in wording only, pure platitudes
  offtopic - a good answer, but to a different question
"""

from __future__ import annotations

import pytest

from app.services.evaluator import Evaluator
from app.services.feedback.judge import NullJudge

pytestmark = pytest.mark.slow

MIN_STRONG_MINUS_OFFTOPIC = 25.0


@pytest.fixture(scope="module")
def evaluator(settings, real_registry) -> Evaluator:
    # NullJudge: this test is about the deterministic scores, not the LLM prose.
    return Evaluator(settings, real_registry, NullJudge())


@pytest.fixture(scope="module")
def scored(evaluator, golden_cases):
    rows = []
    for c in golden_cases:
        row = {"q": c["question"]}
        for tier, text in c.items():
            if tier != "question":
                row[tier] = evaluator.evaluate_text(c["question"], text).score
        rows.append(row)
    return rows


def test_strong_clears_offtopic_by_25(scored):
    bad = [s for s in scored if s["strong"] - s["offtopic"] < MIN_STRONG_MINUS_OFFTOPIC]
    assert not bad, "\n".join(
        f"{s['q']!r}: strong={s['strong']} offtopic={s['offtopic']} "
        f"gap={s['strong'] - s['offtopic']:.1f}"
        for s in bad
    )


def test_tiers_are_ordered(scored):
    bad = [s for s in scored if not (s["strong"] > s["mediocre"] > s["weak"])]
    assert not bad, "\n".join(
        f"{s['q']!r}: strong={s['strong']} mediocre={s['mediocre']} weak={s['weak']}" for s in bad
    )


def test_strong_beats_mediocre_and_weak_clearly(scored):
    bad = [
        s for s in scored if s["strong"] - s["mediocre"] < 8 or s["strong"] - s["weak"] < 15
    ]
    assert not bad, "\n".join(
        f"{s['q']!r}: strong={s['strong']} mediocre={s['mediocre']} weak={s['weak']}"
        for s in bad
    )


def test_offtopic_and_weak_score_low(scored):
    over = [s for s in scored if s["offtopic"] >= 45 or s["weak"] >= 55]
    assert not over, "\n".join(
        f"{s['q']!r}: weak={s['weak']} offtopic={s['offtopic']}" for s in over
    )


def test_strong_answers_land_in_the_good_band(scored):
    # exact calibration of "good" is fuzzy; the floor guards against regression
    # toward the old "everything scores ~70" collapse from the other direction.
    under = [s for s in scored if s["strong"] < 62]
    assert not under, "\n".join(f"{s['q']!r}: strong={s['strong']}" for s in under)


def test_missing_keywords_feedback_is_reachable(evaluator, golden_cases):
    """The old code's missing-keyword branch was mathematically unreachable."""
    case = golden_cases[0]
    result = evaluator.evaluate_text(case["question"], case["weak"])
    assert result.metrics["completeness"].value < 40
    assert result.feedback.improvements  # something concrete to say
