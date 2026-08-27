from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.services.corpus import load_corpus
from app.services.encoders import cosine
from app.services.registry import ModelRegistry

settings = get_settings()
reg = ModelRegistry(settings, load_corpus(settings.corpus_path))
reg.warmup()

cases = [
    json.loads(line)
    for line in (Path(__file__).parents[1] / "tests/data/golden.jsonl").read_text().splitlines()
    if line.strip()
]

for c in cases[:3]:
    q = c["question"]
    print("=" * 80)
    print(q)
    for tier in ("strong", "mediocre", "garbage"):
        ans = c[tier]
        sents = [s.text for s in reg.nlp(ans).sents]
        vecs = np.asarray(reg.encoder.encode([*sents, ans, q], normalize_embeddings=True), np.float32)
        sent_v, ans_v, q_v = vecs[: len(sents)], vecs[-2:-1], vecs[-1:]
        cross = reg.pair_scorer.score_pairs([(q, ans)])[0]
        best_sent = float(cosine(q_v, sent_v).max())
        whole = float(cosine(q_v, ans_v)[0, 0])
        print(f"  {tier:9s} cross={cross:.3f} best_sent_cos={best_sent:.3f} whole_ans_cos={whole:.3f}")
