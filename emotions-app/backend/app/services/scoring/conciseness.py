from __future__ import annotations

import itertools
from collections import Counter

from app.services.scoring.base import MetricOutput, piecewise
from app.services.scoring.text_features import AnswerFeatures

LABEL = "Conciseness"
DETAIL = "Signal-to-noise - substantive content without padding or repetition"


def score_conciseness(features: AnswerFeatures) -> MetricOutput:
    wc = features.word_count
    if wc < 12:
        return MetricOutput(
            value=35.0,
            label=LABEL,
            detail=DETAIL,
            evidence={"reason": "answer too short to assess", "word_count": wc},
        )

    lemmas = features.content_lemmas
    content_ratio = len(lemmas) / wc
    ratio_score = piecewise(
        content_ratio, [(0.2, 35.0), (0.35, 70.0), (0.48, 100.0), (0.62, 92.0), (0.8, 70.0)]
    )

    # Lexical variety: padding circles the same few content words. Low unique /
    # total content-lemma ratio is the clearest "said little, at length" signal.
    unique_ratio = len(set(lemmas)) / max(len(lemmas), 1)
    variety_score = piecewise(
        unique_ratio, [(0.35, 20.0), (0.55, 60.0), (0.75, 95.0), (0.9, 100.0)]
    )

    # exact repeated content bigrams => circling back
    bigrams = Counter(itertools.pairwise(lemmas))
    repeats = sum(c - 1 for c in bigrams.values() if c > 1)
    redundancy = repeats / max(len(lemmas), 1)
    redundancy_score = piecewise(
        redundancy, [(0.0, 100.0), (0.05, 85.0), (0.15, 55.0), (0.3, 25.0)]
    )

    if wc <= 220:
        length_score = 100.0
    elif wc <= 400:
        length_score = 100.0 - (wc - 220) * (30.0 / 180.0)
    else:
        length_score = 60.0

    value = (
        0.3 * ratio_score
        + 0.3 * variety_score
        + 0.2 * redundancy_score
        + 0.2 * length_score
    )
    return MetricOutput(
        value=value,
        label=LABEL,
        detail=DETAIL,
        evidence={
            "content_word_ratio": round(content_ratio, 2),
            "lexical_variety": round(unique_ratio, 2),
            "redundancy": round(redundancy, 3),
            "word_count": wc,
        },
    ).clamped()
