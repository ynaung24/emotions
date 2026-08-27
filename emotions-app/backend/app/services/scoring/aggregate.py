from __future__ import annotations

from app.services.scoring.base import MetricOutput


def aggregate(metrics: dict[str, MetricOutput], weights: dict[str, float]) -> float:
    """Weighted mean over the metrics that were actually computed.

    Weights for absent metrics (e.g. STAR on a non-behavioural question) are
    dropped and the remainder renormalised, so the 0-100 scale is preserved.
    """
    usable = {
        name: m for name, m in metrics.items() if m.computed and weights.get(name, 0) > 0
    }
    if not usable:
        return 0.0

    total_weight = sum(weights[name] for name in usable)
    score = sum(metrics[name].value * weights[name] for name in usable) / total_weight
    return round(score, 1)
