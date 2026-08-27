"""Shared scoring types and the calibration primitive."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field


@dataclass
class MetricOutput:
    value: float  # 0-100
    label: str
    detail: str
    computed: bool = True
    evidence: dict[str, object] = field(default_factory=dict)

    def clamped(self) -> MetricOutput:
        self.value = max(0.0, min(100.0, round(self.value, 1)))
        return self


def piecewise(x: float, anchors: list[tuple[float, float]]) -> float:
    """Linear interpolation through sorted (input, output) anchors, clamped at the ends.

    Replaces the old `(sim + 1) / 2 * 100` which wasted the bottom half of the
    scale (sentence-embedding cosine is effectively non-negative).
    """
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in itertools.pairwise(anchors):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + t * (y1 - y0)
    return anchors[-1][1]


# Anchors were chosen so that off-topic answers land <40 and strong answers >75.
# The golden discrimination test (tests/golden/) is the real guard on these.
RELEVANCE_COSINE = [(0.08, 0.0), (0.22, 28.0), (0.40, 60.0), (0.55, 85.0), (0.70, 100.0)]
KEYWORD_SIM = [(0.15, 0.0), (0.30, 40.0), (0.45, 70.0), (0.60, 95.0), (0.72, 100.0)]
