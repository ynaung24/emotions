"""Text emotion via SamLowe/roberta-base-go_emotions.

Same 28-label taxonomy as the GoEmotions data that was already sitting in the
repo. Runs on the answer text for both text and voice submissions.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.schemas.evaluation import EmotionScore

logger = get_logger(__name__)

_MIN_SCORE = 0.05
_MAX_RETURNED = 6


def classify_emotions(text: str, classifier: object | None) -> list[EmotionScore]:
    if classifier is None or not text.strip():
        return []
    try:
        raw = classifier(text[:2000])  # type: ignore[operator]
    except Exception as exc:
        logger.warning("emotion classification failed: %s", exc)
        return []

    # transformers pipeline with top_k=None returns [[{label, score}, ...]]
    rows = raw[0] if raw and isinstance(raw[0], list) else raw
    scored = sorted(
        (EmotionScore(label=r["label"], score=float(r["score"])) for r in rows),
        key=lambda e: e.score,
        reverse=True,
    )
    kept = [e for e in scored if e.score >= _MIN_SCORE][:_MAX_RETURNED]
    return kept or scored[:1]
