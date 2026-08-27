from __future__ import annotations

import statistics

import textstat

from app.services.scoring.base import MetricOutput, piecewise
from app.services.scoring.text_features import AnswerFeatures

LABEL = "Clarity"
DETAIL = "Readability and sentence structure - not too dense, not choppy, some variety"

_FLESCH = [(0.0, 25.0), (30.0, 60.0), (50.0, 85.0), (65.0, 100.0), (80.0, 92.0), (100.0, 70.0)]


def score_clarity(features: AnswerFeatures) -> MetricOutput:
    text = features.text
    lengths = features.sentence_lengths

    try:
        flesch = float(textstat.flesch_reading_ease(text))
    except Exception:
        flesch = 50.0
    flesch_score = piecewise(flesch, _FLESCH)

    mean_len = statistics.fmean(lengths) if lengths else 0.0
    if mean_len <= 0:
        length_score = 0.0
    elif mean_len < 8:
        length_score = 55.0 + (mean_len / 8) * 30.0
    elif mean_len <= 22:
        length_score = 100.0
    elif mean_len <= 34:
        length_score = 100.0 - (mean_len - 22) * (35.0 / 12.0)
    else:
        length_score = 55.0

    # Some variation reads better than a monotone wall of same-length sentences.
    if len(lengths) >= 3 and mean_len > 0:
        cv = statistics.pstdev(lengths) / mean_len
        variety_score = piecewise(cv, [(0.0, 70.0), (0.35, 100.0), (0.9, 88.0), (1.6, 60.0)])
    else:
        variety_score = 80.0

    value = 0.45 * flesch_score + 0.4 * length_score + 0.15 * variety_score
    return MetricOutput(
        value=value,
        label=LABEL,
        detail=DETAIL,
        evidence={
            "flesch_reading_ease": round(flesch, 1),
            "mean_sentence_length": round(mean_len, 1),
            "sentence_count": len(lengths),
        },
    ).clamped()
