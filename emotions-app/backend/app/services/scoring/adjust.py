"""Cross-metric adjustments applied by the orchestrator after individual scoring.

These encode two things a single metric can't see on its own:

1. A topical-but-empty answer scores high on bi-encoder relevance (it echoes the
   question's vocabulary) while covering none of the substance. Damp relevance by
   how much substance is actually present.
2. A well-structured behavioural answer *is* a relevant answer even when it never
   restates the question - floor its relevance with its STAR score.
"""

from __future__ import annotations


def damp_relevance(relevance: float, completeness: float) -> float:
    """Scale relevance from 65%-100% of its value based on substance coverage.

    A topical-but-empty answer (high relevance, near-zero completeness) loses
    about a third of its relevance; a substantive answer keeps all of it.
    """
    factor = 0.65 + 0.35 * min(1.0, completeness / 55.0)
    return relevance * factor


def behavioural_relevance_floor(relevance: float, structure: float) -> float:
    """A well-structured STAR answer is relevant even if it never restates the question."""
    return max(relevance, 0.9 * structure)


def behavioural_completeness_floor(completeness: float, structure: float) -> float:
    """Behavioural corpus keywords are abstract STAR meta-vocabulary ('obstacle',
    'setback') that natural answers rarely say verbatim; the STAR arc itself is
    the better completeness signal for these."""
    return max(completeness, 0.75 * structure)
