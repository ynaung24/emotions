"""Deterministic feedback - always runs, no network. Driven by real metric evidence.

The old code had two keyword-feedback branches that were mathematically unreachable
(`score < 0.5` after a `(sim+1)/2` rescale). This version reads the actual
covered/missing lists the new completeness scorer produces.
"""

from __future__ import annotations

from app.schemas.evaluation import FeedbackBlock
from app.services.scoring.base import MetricOutput

_STRONG = 75.0
_WEAK = 58.0


def _band(score: float) -> str:
    if score >= 82:
        return "This is a strong answer."
    if score >= 68:
        return "This is a solid answer with a few clear places to tighten up."
    if score >= 50:
        return "This answer is on topic but underdeveloped."
    return "This answer does not yet address the question effectively."


def build_template_feedback(
    overall: float, metrics: dict[str, MetricOutput]
) -> FeedbackBlock:
    strengths: list[str] = []
    improvements: list[str] = []

    for name, m in metrics.items():
        if not m.computed:
            continue
        ev = m.evidence
        if m.value >= _STRONG:
            strengths.append(_strength_line(name, m, ev))
        elif m.value < _WEAK:
            improvements.append(_improvement_line(name, m, ev))

    computed = [(n, m) for n, m in metrics.items() if m.computed]

    if not strengths and computed:
        best = max(computed, key=lambda nm: nm[1].value)[1]
        strengths.append(
            f"{best.label} is the strongest dimension here ({best.value:.0f}/100)."
        )

    if not improvements and computed:
        # even a good answer has a weakest link worth naming
        name, weakest = min(computed, key=lambda nm: nm[1].value)
        if weakest.value < 85:
            improvements.append(_improvement_line(name, weakest, weakest.evidence))

    return FeedbackBlock(
        summary=_band(overall),
        strengths=strengths[:4],
        improvements=improvements[:4],
        source="template",
    )


def _items(ev: dict[str, object], key: str, limit: int) -> list[str]:
    raw = ev.get(key)
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(x) for x in raw][:limit]


def _strength_line(name: str, m: MetricOutput, ev: dict[str, object]) -> str:
    if name == "completeness" and _items(ev, "covered", 4):
        return f"Good coverage of the key points ({', '.join(_items(ev, 'covered', 4))})."
    if name == "specificity" and _items(ev, "named", 3):
        joined = ", ".join(_items(ev, "named", 3))
        return f"Concrete and specific - you name real examples ({joined})."
    if name == "structure":
        return "Clear STAR structure - situation, action and result all come through."
    return f"{m.label} is strong ({m.value:.0f}/100)."


def _improvement_line(name: str, m: MetricOutput, ev: dict[str, object]) -> str:
    if name == "completeness" and _items(ev, "missing", 4):
        return f"Missing points a strong answer would cover: {', '.join(_items(ev, 'missing', 4))}."
    if name == "relevance":
        return "Parts of the answer drift from what was asked - keep it anchored to the question."
    if name == "clarity":
        ml = ev.get("mean_sentence_length")
        if isinstance(ml, (int, float)) and ml > 26:
            return "Sentences run long - break them up so each carries one idea."
        return "Structure the answer into clearer, self-contained points."
    if name == "specificity":
        return "Too general - add a concrete example, a tool you used, or a number."
    if name == "confidence":
        hp = ev.get("hedges_per_sentence")
        if isinstance(hp, (int, float)) and hp >= 0.5:
            return (
                "Frequent hedging ('I think', 'maybe') undercuts the answer - "
                "state things directly."
            )
        return "Phrase your points more assertively."
    if name == "conciseness":
        return "Trim repetition and filler so the substance stands out."
    if name == "structure" and _items(ev, "components_missing", 4):
        miss = ", ".join(_items(ev, "components_missing", 4))
        return f"Behavioural answers need the full STAR arc - you're missing: {miss}."
    return f"{m.label} needs work ({m.value:.0f}/100)."
