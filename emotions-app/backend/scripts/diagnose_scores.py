"""Print the full metric breakdown for every golden case - tuning aid, not a test."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import get_settings
from app.services.corpus import load_corpus
from app.services.evaluator import Evaluator
from app.services.feedback.judge import NullJudge
from app.services.registry import ModelRegistry

settings = get_settings()
corpus = load_corpus(settings.corpus_path)
reg = ModelRegistry(settings, corpus)
reg.warmup()
ev = Evaluator(settings, reg, NullJudge())

cases = [
    json.loads(line)
    for line in (Path(__file__).parents[1] / "tests/data/golden.jsonl").read_text().splitlines()
    if line.strip()
]

for c in cases:
    print("=" * 90)
    print(c["question"])
    for tier in ("strong", "mediocre", "weak", "offtopic"):
        if tier not in c:
            continue
        r = ev.evaluate_text(c["question"], c[tier])
        parts = " ".join(f"{k}={v.value:.0f}" for k, v in r.metrics.items())
        gen = " [generic]" if r.scored_generically else ""
        print(f"  {tier:9s} {r.score:5.1f}{gen}  | {parts}")
