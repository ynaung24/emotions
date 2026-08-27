from __future__ import annotations

from app.services.scoring.base import MetricOutput, piecewise
from app.services.scoring.text_features import AnswerFeatures

LABEL = "Confidence"
DETAIL = "Assertive phrasing - low on hedging ('I think', 'maybe') and filler"


def score_confidence(
    features: AnswerFeatures, prosody_steadiness: float | None = None
) -> MetricOutput:
    sentences = max(features.sentence_count, 1)
    per_100w = 100.0 / max(features.word_count, 1)

    hedge_density = features.hedge_count / sentences
    filler_density = features.filler_count * per_100w

    hedge_score = piecewise(hedge_density, [(0.0, 100.0), (0.4, 82.0), (1.0, 55.0), (2.0, 25.0)])
    filler_score = piecewise(filler_density, [(0.0, 100.0), (1.5, 80.0), (4.0, 50.0), (8.0, 20.0)])

    value = 0.6 * hedge_score + 0.4 * filler_score
    evidence: dict[str, object] = {
        "hedges_per_sentence": round(hedge_density, 2),
        "fillers_per_100w": round(filler_density, 2),
    }

    if prosody_steadiness is not None:
        voice_score = piecewise(prosody_steadiness, [(0.0, 40.0), (0.5, 75.0), (1.0, 100.0)])
        value = 0.7 * value + 0.3 * voice_score
        evidence["prosody_steadiness"] = round(prosody_steadiness, 2)

    return MetricOutput(value=value, label=LABEL, detail=DETAIL, evidence=evidence).clamped()
