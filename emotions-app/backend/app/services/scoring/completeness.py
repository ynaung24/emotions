from __future__ import annotations

import re

import numpy as np

from app.services.encoders import cosine
from app.services.scoring.base import KEYWORD_SIM, MetricOutput, piecewise

LABEL = "Completeness"
DETAIL = "How many of the points a strong answer would cover are present"

_STOP = {"a", "an", "the", "of", "and", "or", "to", "in", "for", "with", "on"}


def _keyword_tokens(keyword: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", keyword.lower()) if t and t not in _STOP}


def score_completeness(
    keywords: tuple[str, ...],
    keyword_vecs: np.ndarray,
    sentence_vecs: np.ndarray,
    answer_vocab: frozenset[str],
    hit_threshold: float = 0.5,
) -> MetricOutput:
    if not keywords or keyword_vecs.shape[0] == 0 or sentence_vecs.size == 0:
        return MetricOutput(
            value=0.0,
            label=LABEL,
            detail=DETAIL,
            computed=False,
            evidence={"reason": "no reference keywords available"},
        )

    sem = cosine(keyword_vecs, sentence_vecs).max(axis=1)  # best sentence per keyword

    per_keyword: list[float] = []
    covered: list[str] = []
    missing: list[str] = []
    for kw, sem_sim in zip(keywords, sem, strict=True):
        toks = _keyword_tokens(kw)
        lex = len(toks & answer_vocab) / len(toks) if toks else 0.0
        # literal mention is worth full credit; semantic match backfills partial
        score = max(lex, piecewise(float(sem_sim), KEYWORD_SIM) / 100.0 * 0.9)
        per_keyword.append(score)
        (covered if score >= hit_threshold else missing).append(kw)

    coverage_ratio = len(covered) / len(keywords)
    mean_score = float(np.mean(per_keyword)) * 100.0
    # keyword lists are aspirational - a strong 45-second answer realistically
    # covers 40-55% of them, so the curve tops out well before 100%.
    coverage_score = piecewise(
        coverage_ratio, [(0.0, 0.0), (0.2, 45.0), (0.4, 78.0), (0.55, 95.0), (0.75, 100.0)]
    )

    value = 0.5 * mean_score + 0.5 * coverage_score
    return MetricOutput(
        value=value,
        label=LABEL,
        detail=DETAIL,
        evidence={
            "covered": covered,
            "missing": missing,
            "coverage_ratio": round(coverage_ratio, 2),
        },
    ).clamped()
