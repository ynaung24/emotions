from __future__ import annotations

import numpy as np

from app.services.encoders import cosine
from app.services.scoring.base import RELEVANCE_COSINE, MetricOutput, piecewise

LABEL = "Relevance"
DETAIL = "How directly the answer addresses the question asked"


def score_relevance(
    question: str,
    answer: str,
    question_vec: np.ndarray,
    answer_vec: np.ndarray,
    sentence_vecs: np.ndarray,
) -> MetricOutput:
    """Bi-encoder cosine only.

    An earlier version blended in an ms-marco cross-encoder; on adversarial
    'on-topic but empty' answers it inverted (platitudes that echoed the question
    wording scored ~0.98, real procedural answers ~0.01), so it was removed.
    Bi-encoder cosine reliably separates genuinely off-topic answers (~0.0-0.1)
    from on-topic ones (~0.4-0.7); the substance metrics (completeness,
    specificity, structure) do the finer discrimination.
    """
    if sentence_vecs.size:
        sims = cosine(question_vec, sentence_vecs)[0]
        best = float(np.max(sims))
        mean = float(np.mean(sims))
    else:
        best = mean = 0.0
    whole = float(cosine(question_vec, answer_vec)[0, 0])

    signal = 0.45 * best + 0.35 * whole + 0.20 * mean
    value = piecewise(signal, RELEVANCE_COSINE)
    return MetricOutput(
        value=value,
        label=LABEL,
        detail=DETAIL,
        evidence={
            "best_sentence_cosine": round(best, 3),
            "whole_answer_cosine": round(whole, 3),
            "mean_sentence_cosine": round(mean, 3),
        },
    ).clamped()
