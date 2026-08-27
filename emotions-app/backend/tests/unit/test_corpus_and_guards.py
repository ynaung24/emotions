from __future__ import annotations

import json

import pytest

from app.services.corpus import CorpusError, load_corpus
from app.services.guards import contains_inappropriate_content
from app.services.scoring.aggregate import aggregate
from app.services.scoring.base import MetricOutput


def test_corpus_loads_and_normalises_weights(corpus):
    assert len(corpus.questions) == 20
    assert abs(sum(corpus.weights.values()) - 1.0) < 1e-6
    assert corpus.by_id("q4").is_behavioral


def test_corpus_rejects_missing_file(tmp_path):
    with pytest.raises(CorpusError):
        load_corpus(tmp_path / "nope.json")


def test_corpus_rejects_bad_shape(tmp_path):
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps({"questions": []}))
    with pytest.raises(CorpusError):
        load_corpus(p)


def test_guard_exact_word_only():
    assert contains_inappropriate_content("this is shit", {"shit"})
    assert not contains_inappropriate_content("classification", {"ass"})


def test_aggregate_drops_uncomputed_and_renormalises():
    metrics = {
        "relevance": MetricOutput(80, "Relevance", "", computed=True),
        "structure": MetricOutput(0, "Structure", "", computed=False),
    }
    weights = {"relevance": 0.5, "structure": 0.5}
    # structure absent -> relevance carries the whole score
    assert aggregate(metrics, weights) == 80.0
