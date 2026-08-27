from __future__ import annotations

from app.services.scoring.base import MetricOutput, piecewise
from app.services.scoring.text_features import AnswerFeatures

LABEL = "Specificity"
DETAIL = "Concrete detail - named tools, numbers, examples rather than generalities"

# Entity labels that signal a concrete anecdote (skip DATE/TIME/CARDINAL noise).
_CONCRETE_LABELS = {"ORG", "PRODUCT", "PERSON", "GPE", "WORK_OF_ART", "EVENT", "LANGUAGE", "FAC"}


def score_specificity(features: AnswerFeatures) -> MetricOutput:
    wc = max(features.word_count, 1)
    per100 = 100.0 / wc

    concrete_ents = [e for e, label in features.entities if label in _CONCRETE_LABELS]
    named = {*concrete_ents, *features.proper_nouns}
    named_density = len(named) * per100
    num_density = features.numeral_count * per100

    unique_content = len(set(features.content_lemmas))
    lexical_variety = unique_content / wc

    named_score = piecewise(
        named_density, [(0.0, 22.0), (1.0, 55.0), (2.5, 82.0), (5.0, 100.0)]
    )
    num_score = piecewise(num_density, [(0.0, 45.0), (1.0, 72.0), (3.0, 95.0), (5.0, 100.0)])
    # a short answer has high type-token ratio for free; scale variety by length
    variety_score = piecewise(
        lexical_variety, [(0.15, 30.0), (0.3, 68.0), (0.45, 92.0), (0.6, 100.0)]
    ) * min(1.0, wc / 45.0)

    value = 0.62 * named_score + 0.22 * num_score + 0.16 * variety_score
    return MetricOutput(
        value=value,
        label=LABEL,
        detail=DETAIL,
        evidence={
            "named": sorted(named)[:12],
            "named_per_100w": round(named_density, 2),
            "numerals_per_100w": round(num_density, 2),
        },
    ).clamped()
