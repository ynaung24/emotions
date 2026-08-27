"""Owns every heavy object in the process and builds each one exactly once.

The old code constructed the sentence-transformer twice (module import + a second
call in main) and re-read corpus.json on every request. Here everything is built
in `warmup()` during the app lifespan, and per-corpus embeddings are precomputed.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.corpus import Corpus, Question
from app.services.encoders import TextEncoder, cosine

logger = get_logger(__name__)

STAR_PROTOTYPES: dict[str, str] = {
    "situation": "I was in a situation where the context and background was as follows.",
    "task": "My responsibility and the goal I needed to achieve was this.",
    "action": "The specific steps I personally took were these.",
    "result": "The outcome, impact, and what I learned was this.",
}


def _enriched_text(q: Question) -> str:
    # The expected talking points alone - deliberately NOT the question wording,
    # so an answer that merely parrots the question ("evaluating performance is
    # important") doesn't score as relevant; it has to be about the substance.
    if q.expected_keywords:
        return f"{q.category}: {', '.join(q.expected_keywords)}"
    return q.text


class ModelRegistry:
    def __init__(self, settings: Settings, corpus: Corpus) -> None:
        self.settings = settings
        self.corpus = corpus
        self._lock = threading.Lock()
        self._built: Counter[str] = Counter()

        # SentenceTransformer satisfies TextEncoder structurally but mypy can't
        # confirm it against the untyped signature - hold as Any.
        self._encoder: Any = None
        self._nlp: object | None = None
        self._transcriber: object | None = None
        self._emotion: object | None = None

        self._question_vecs: np.ndarray | None = None
        self._keyword_vecs: dict[str, np.ndarray] = {}
        self._enriched_vecs: dict[str, np.ndarray] = {}
        self._star_vecs: np.ndarray | None = None

    # ------------------------------------------------------------------ builders

    @property
    def encoder(self) -> TextEncoder:
        if self._encoder is None:
            with self._lock:
                if self._encoder is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("loading bi-encoder %s", self.settings.bi_encoder_model)
                    self._encoder = SentenceTransformer(
                        self.settings.bi_encoder_model,
                        cache_folder=str(self.settings.model_cache_dir),
                    )
                    self._built["encoder"] += 1
        return self._encoder

    @property
    def nlp(self) -> object:
        if self._nlp is None:
            with self._lock:
                if self._nlp is None:
                    import spacy

                    logger.info("loading spaCy %s", self.settings.spacy_model)
                    # Keep parser (sentence boundaries), tagger, lemmatizer, NER.
                    self._nlp = spacy.load(self.settings.spacy_model, disable=["textcat"])
                    self._built["nlp"] += 1
        return self._nlp

    @property
    def transcriber(self) -> object:
        if self._transcriber is None:
            with self._lock:
                if self._transcriber is None:
                    from faster_whisper import WhisperModel

                    logger.info("loading whisper %s", self.settings.whisper_model)
                    self._transcriber = WhisperModel(
                        self.settings.whisper_model,
                        device="cpu",
                        compute_type=self.settings.whisper_compute_type,
                        download_root=str(self.settings.model_cache_dir),
                    )
                    self._built["transcriber"] += 1
        return self._transcriber

    @property
    def emotion_classifier(self) -> object | None:
        if not self.settings.enable_emotion:
            return None
        if self._emotion is None:
            with self._lock:
                if self._emotion is None:
                    from transformers import pipeline

                    logger.info("loading emotion model %s", self.settings.emotion_model)
                    self._emotion = pipeline(
                        "text-classification",
                        model=self.settings.emotion_model,
                        top_k=None,
                        truncation=True,
                    )
                    self._built["emotion"] += 1
        return self._emotion

    # --------------------------------------------------------------- precompute

    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.encoder.encode(texts, normalize_embeddings=True), dtype=np.float32
        )

    def precompute(self) -> None:
        if self._question_vecs is None:
            self._question_vecs = self._embed([q.text for q in self.corpus.questions])
        for q in self.corpus.questions:
            if q.id not in self._keyword_vecs and q.expected_keywords:
                self._keyword_vecs[q.id] = self._embed(list(q.expected_keywords))
            if q.id not in self._enriched_vecs:
                self._enriched_vecs[q.id] = self._embed([_enriched_text(q)])[0]
        if self._star_vecs is None:
            self._star_vecs = self._embed(list(STAR_PROTOTYPES.values()))

    def relevance_target(self, question: Question) -> np.ndarray:
        """Embedding of the question plus its expected talking points.

        Open-ended prompts ('tell me about yourself') carry little signal on their
        own; anchoring relevance to the substance the question is after keeps a
        specific, on-point answer from being marked irrelevant.
        """
        vec = self._enriched_vecs.get(question.id)
        if vec is None:
            vec = self._embed([_enriched_text(question)])[0]
            self._enriched_vecs[question.id] = vec
        return vec.reshape(1, -1)

    @property
    def star_vectors(self) -> np.ndarray:
        if self._star_vecs is None:
            self.precompute()
        assert self._star_vecs is not None
        return self._star_vecs

    def keyword_vectors(self, question: Question) -> tuple[tuple[str, ...], np.ndarray]:
        """Curated keyword embeddings for a corpus question, or an empty pair."""
        if question.id in self._keyword_vecs:
            return question.expected_keywords, self._keyword_vecs[question.id]
        if question.expected_keywords:
            vecs = self._embed(list(question.expected_keywords))
            self._keyword_vecs[question.id] = vecs
            return question.expected_keywords, vecs
        return (), np.zeros((0, 1), dtype=np.float32)

    # ------------------------------------------------------------------- match

    def match_question(self, text: str) -> tuple[Question | None, float]:
        """Best corpus question by embedding cosine. Enrichment, not a gate -
        callers still score when this returns (None, low)."""
        if self._question_vecs is None:
            self.precompute()
        assert self._question_vecs is not None
        query = self._embed([text])
        sims = cosine(query, self._question_vecs)[0]
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        if best >= self.settings.question_match_threshold:
            return self.corpus.questions[idx], best
        return None, best

    # ------------------------------------------------------------------ status

    def warmup(self) -> None:
        started = time.perf_counter()
        _ = self.encoder
        _ = self.nlp
        self.precompute()
        # whisper + emotion are lazy on first use; they are not on the text path.
        logger.info("warmup complete in %.1fs", time.perf_counter() - started)

    def status(self) -> dict[str, bool]:
        return {
            "encoder": self._encoder is not None,
            "nlp": self._nlp is not None,
            "precomputed": self._question_vecs is not None,
        }

    @property
    def build_counts(self) -> dict[str, int]:
        return dict(self._built)

    @property
    def ready(self) -> bool:
        return all(self.status().values())
