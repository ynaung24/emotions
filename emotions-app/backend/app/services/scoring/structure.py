from __future__ import annotations

import itertools

import numpy as np

from app.services.encoders import cosine
from app.services.scoring.base import MetricOutput

LABEL = "Structure (STAR)"
DETAIL = "Situation / Task / Action / Result arc - expected for behavioural answers"

_COMPONENTS = ("situation", "task", "action", "result")
_PRESENCE_SIM = 0.28


def score_structure(sentence_vecs: np.ndarray, star_vecs: np.ndarray) -> MetricOutput:
    if sentence_vecs.size == 0 or star_vecs.shape[0] != 4:
        return MetricOutput(
            value=0.0, label=LABEL, detail=DETAIL, computed=False,
            evidence={"reason": "not enough content"},
        )

    sim = cosine(star_vecs, sentence_vecs)  # (4, n_sentences)
    best_sim = sim.max(axis=1)
    best_idx = sim.argmax(axis=1)

    present = {c: float(best_sim[i]) >= _PRESENCE_SIM for i, c in enumerate(_COMPONENTS)}
    n_present = sum(present.values())
    presence_score = n_present / 4 * 100.0

    # rough ordering: indices of present components should trend upward
    order_bonus = 0.0
    present_positions = [int(best_idx[i]) for i, c in enumerate(_COMPONENTS) if present[c]]
    if len(present_positions) >= 2:
        ascending = sum(
            1 for a, b in itertools.pairwise(present_positions) if b >= a
        )
        order_bonus = (ascending / (len(present_positions) - 1)) * 15.0

    value = min(100.0, presence_score * 0.85 + order_bonus)
    return MetricOutput(
        value=value,
        label=LABEL,
        detail=DETAIL,
        evidence={
            "components_present": [c for c in _COMPONENTS if present[c]],
            "components_missing": [c for c in _COMPONENTS if not present[c]],
        },
    ).clamped()
