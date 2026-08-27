"""API-level integration tests against the real app (models load once)."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:  # triggers lifespan -> warmup
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/ready").json()
    assert body["ready"] is True


def test_questions_returns_objects(client):
    body = client.get("/questions").json()
    assert len(body["questions"]) == 20
    first = body["questions"][0]
    assert set(first) == {"id", "text", "category"}


def test_evaluate_text_discriminates(client):
    q = "How do you evaluate model performance?"
    strong = client.post(
        "/evaluate/text",
        json={
            "question": q,
            "answer": (
                "I use a confusion matrix, then precision, recall and F1 per class, and "
                "the precision-recall curve with AUC when classes are imbalanced. I "
                "evaluate on a held-out test set matching production and monitor drift."
            ),
        },
    ).json()
    weak = client.post(
        "/evaluate/text",
        json={"question": q, "answer": "I check if the accuracy is high enough and move on."},
    ).json()

    assert strong["score"] > weak["score"] + 15
    assert "conciseness" in strong["metrics"]
    assert strong["metrics"]["conciseness"]["computed"] is True


def test_evaluate_text_schema_has_transcript_field(client):
    r = client.post(
        "/evaluate/text",
        json={
            "question": "Tell me about yourself.",
            "answer": "I am a data scientist with five years of experience building models.",
        },
    )
    assert r.status_code == 200
    assert "transcript" in r.json()  # was silently stripped by the old response_model


def test_unknown_question_still_scores(client):
    r = client.post(
        "/evaluate/text",
        json={
            "question": (
                "What is your favourite kind of database index and when would you avoid one?"
            ),
            "answer": (
                "I reach for a B-tree index on high-cardinality columns I filter or join on, "
                "but I avoid indexing columns with very low cardinality or tables with heavy "
                "write throughput where the index maintenance cost outweighs read gains."
            ),
        },
    ).json()
    assert r["scored_generically"] is True
    assert r["score"] > 0
    assert "scored_generically" in r["flags"]


def test_openapi_schema_snapshot(client):
    schema = client.get("/openapi.json").json()
    result = schema["components"]["schemas"]["EvaluationResult"]["properties"]
    for field in ("score", "metrics", "feedback", "transcript", "scored_generically", "emotions"):
        assert field in result


def _silent_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.zeros(16000, dtype=np.int16)).tobytes())
    return buf.getvalue()


def test_evaluate_audio_silence_is_flagged(client):
    r = client.post(
        "/evaluate/audio",
        files={"file": ("s.wav", _silent_wav(), "audio/wav")},
        data={"question": "Tell me about yourself."},
    )
    assert r.status_code == 200
    assert "no_speech_detected" in r.json()["flags"]
