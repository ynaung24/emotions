"""Load and validate the evaluation corpus.

Kept deliberately free of any ML dependency - embeddings over this data live in
`ModelRegistry`. This is just parsing + validation, adapted from the validation
block of the old `evaluate.load_corpus()`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Criteria the scorer produces beyond what corpus.json historically weighted.
# corpus.json weights are honoured; anything missing falls back to these.
#
# The discriminating signals (relevance, completeness, specificity, structure)
# carry the weight. clarity/confidence/conciseness sit near ~90 for almost any
# coherent answer, so they act as tie-breakers, not drivers - keeping their
# weight low is what stops the "every answer scores ~70" failure mode.
DEFAULT_WEIGHTS: dict[str, float] = {
    "relevance": 0.34,
    "completeness": 0.30,
    "specificity": 0.17,
    "clarity": 0.07,
    "confidence": 0.06,
    "conciseness": 0.06,
    "structure": 0.14,  # only applied when present (behavioural questions)
}

BEHAVIORAL_CATEGORIES = {"behavioral", "project", "communication"}


class CorpusError(ValueError):
    """Raised when corpus.json is missing or structurally invalid."""


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    category: str
    expected_keywords: tuple[str, ...]

    @property
    def is_behavioral(self) -> bool:
        return self.category in BEHAVIORAL_CATEGORIES


@dataclass(frozen=True)
class Corpus:
    questions: tuple[Question, ...]
    weights: dict[str, float]
    inappropriate_words: frozenset[str]
    criteria_descriptions: dict[str, str] = field(default_factory=dict)

    def by_id(self, qid: str) -> Question | None:
        return next((q for q in self.questions if q.id == qid), None)


def load_corpus(path: Path) -> Corpus:
    if not path.exists():
        raise CorpusError(f"corpus file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"corpus.json is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise CorpusError("corpus.json must be a JSON object")
    if not isinstance(raw.get("questions"), list) or not raw["questions"]:
        raise CorpusError("corpus.json must contain a non-empty 'questions' list")

    questions: list[Question] = []
    for i, q in enumerate(raw["questions"]):
        if not isinstance(q, dict) or "text" not in q:
            raise CorpusError(f"question at index {i} is missing a 'text' field")
        questions.append(
            Question(
                id=str(q.get("id") or f"q{i + 1}"),
                text=str(q["text"]).strip(),
                category=str(q.get("category", "general")),
                expected_keywords=tuple(
                    str(k).strip() for k in q.get("expected_keywords", []) if str(k).strip()
                ),
            )
        )

    weights = dict(DEFAULT_WEIGHTS)
    descriptions: dict[str, str] = {}
    criteria = raw.get("evaluation_criteria", {})
    if isinstance(criteria, dict):
        for name, spec in criteria.items():
            if isinstance(spec, dict) and isinstance(spec.get("weight"), (int, float)):
                weights[name] = float(spec["weight"])
            if isinstance(spec, dict) and spec.get("description"):
                descriptions[name] = str(spec["description"])
    weights = _normalise(weights)

    words = raw.get("inappropriate_words", [])
    inappropriate = frozenset(str(w).lower().strip() for w in words if str(w).strip())

    return Corpus(
        questions=tuple(questions),
        weights=weights,
        inappropriate_words=inappropriate,
        criteria_descriptions=descriptions,
    )


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    total = sum(v for v in weights.values() if v > 0)
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in weights.items() if v > 0}
