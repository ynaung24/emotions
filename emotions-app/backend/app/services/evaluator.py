"""Orchestrates a single evaluation: one spaCy pass, one encoder pass, then the
pure scoring functions, aggregation, and feedback.
"""

from __future__ import annotations

import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.evaluation import (
    DeliveryMetrics,
    EvaluationResult,
    FeedbackBlock,
    MetricResult,
    QuestionOut,
)
from app.services.corpus import Question
from app.services.emotion import classify_emotions
from app.services.feedback.judge import Judge, JudgeContext
from app.services.feedback.templates import build_template_feedback
from app.services.guards import contains_inappropriate_content
from app.services.registry import ModelRegistry
from app.services.scoring.adjust import (
    behavioural_completeness_floor,
    behavioural_relevance_floor,
    damp_relevance,
)
from app.services.scoring.aggregate import aggregate
from app.services.scoring.base import MetricOutput
from app.services.scoring.clarity import score_clarity
from app.services.scoring.completeness import score_completeness
from app.services.scoring.conciseness import score_conciseness
from app.services.scoring.confidence import score_confidence
from app.services.scoring.relevance import score_relevance
from app.services.scoring.specificity import score_specificity
from app.services.scoring.structure import score_structure
from app.services.scoring.text_features import extract_features
from app.services.transcription import transcribe

logger = get_logger(__name__)

_GENERIC_WORDS = {
    "time", "example", "situation", "experience", "way", "thing", "someone",
    "something", "day", "moment", "instance", "story", "yourself", "one",
}


class Evaluator:
    def __init__(self, settings: Settings, registry: ModelRegistry, judge: Judge) -> None:
        self.settings = settings
        self.registry = registry
        self.judge = judge

    # ---------------------------------------------------------------- public

    def evaluate_text(
        self, question: str, answer: str, role: str | None = None
    ) -> EvaluationResult:
        return self._evaluate(question, answer, transcript=None, delivery=None, prosody=None)

    def evaluate_audio(self, question: str, raw_audio: bytes) -> EvaluationResult:
        tr = transcribe(raw_audio, self.registry.transcriber)
        if not tr.text.strip():
            return EvaluationResult(
                score=0.0,
                metrics={},
                feedback=FeedbackBlock(
                    summary="No speech could be transcribed from the recording.",
                    improvements=["Record again in a quiet room and speak clearly."],
                ),
                transcript="",
                flags=["no_speech_detected"],
            )
        from app.services.delivery import analyse_delivery

        delivery, steadiness = analyse_delivery(tr)
        return self._evaluate(
            question, tr.text, transcript=tr.text, delivery=delivery, prosody=steadiness
        )

    # --------------------------------------------------------------- internal

    def _evaluate(
        self,
        question: str,
        answer: str,
        *,
        transcript: str | None,
        delivery: DeliveryMetrics | None,
        prosody: float | None,
    ) -> EvaluationResult:
        corpus = self.registry.corpus

        if contains_inappropriate_content(answer, corpus.inappropriate_words):
            return EvaluationResult(
                score=0.0,
                metrics={},
                feedback=FeedbackBlock(
                    summary="This response contains inappropriate language.",
                    improvements=["Keep interview answers professional in tone."],
                ),
                transcript=transcript,
                delivery=delivery,
                flags=["inappropriate_content"],
            )

        reg = self.registry
        matched_q, _match_score = reg.match_question(question)

        features = extract_features(answer, reg.nlp)
        sentences = features.sentences or [answer]

        # single bi-encoder pass: every sentence, the whole answer, the question
        packed = np.asarray(
            reg.encoder.encode([*sentences, answer, question], normalize_embeddings=True),
            dtype=np.float32,
        )
        sentence_vecs = packed[: len(sentences)]
        answer_vec = packed[len(sentences) : len(sentences) + 1]
        question_vec = packed[len(sentences) + 1 : len(sentences) + 2]

        if matched_q is not None:
            keywords, keyword_vecs = reg.keyword_vectors(matched_q)
        else:
            keywords = self._pseudo_keywords(question)
            keyword_vecs = (
                np.asarray(
                    reg.encoder.encode(list(keywords), normalize_embeddings=True), np.float32
                )
                if keywords
                else np.zeros((0, 1), dtype=np.float32)
            )

        scores: dict[str, MetricOutput] = {
            "relevance": score_relevance(
                question, answer, question_vec, answer_vec, sentence_vecs
            ),
            "completeness": score_completeness(
                keywords,
                keyword_vecs,
                sentence_vecs,
                features.token_vocab,
                self.settings.keyword_hit_threshold,
            ),
            "clarity": score_clarity(features),
            "specificity": score_specificity(features),
            "confidence": score_confidence(features, prosody),
            "conciseness": score_conciseness(features),
        }
        if matched_q is not None and matched_q.is_behavioral:
            structure = score_structure(sentence_vecs, reg.star_vectors)
            scores["structure"] = structure
            if structure.computed:
                scores["relevance"].value = behavioural_relevance_floor(
                    scores["relevance"].value, structure.value
                )
                scores["completeness"].value = behavioural_completeness_floor(
                    scores["completeness"].value, structure.value
                )

        # topical-but-empty answers over-score on relevance; damp by substance
        raw_relevance = scores["relevance"].value
        scores["relevance"].value = round(
            damp_relevance(raw_relevance, scores["completeness"].value), 1
        )
        scores["relevance"].evidence["raw_value"] = round(raw_relevance, 1)

        overall = aggregate(scores, corpus.weights)

        metrics = {name: _to_result(m) for name, m in scores.items()}
        feedback = self._feedback(question, answer, overall, scores, transcript)

        emotions = classify_emotions(answer, reg.emotion_classifier)

        flags: list[str] = []
        if matched_q is None:
            flags.append("scored_generically")

        return EvaluationResult(
            score=overall,
            metrics=metrics,
            feedback=feedback,
            matched_question=_question_out(matched_q),
            scored_generically=matched_q is None,
            emotions=emotions,
            transcript=transcript,
            delivery=delivery,
            flags=flags,
        )

    def _feedback(
        self,
        question: str,
        answer: str,
        overall: float,
        scores: dict[str, MetricOutput],
        transcript: str | None,
    ) -> FeedbackBlock:
        template = build_template_feedback(overall, scores)
        judged = self.judge.generate(
            JudgeContext(
                question=question,
                answer=answer,
                overall_score=overall,
                metrics=scores,
                transcript=transcript,
            )
        )
        if judged is None:
            return template
        # keep template bullets as a floor if the judge was terse
        if not judged.strengths:
            judged.strengths = template.strengths
        if not judged.improvements:
            judged.improvements = template.improvements
        return judged

    def _pseudo_keywords(self, question: str) -> tuple[str, ...]:
        """Derive reference points from the question itself when it isn't in the corpus."""
        doc = self.registry.nlp(question)  # type: ignore[operator]
        out: list[str] = []
        for chunk in doc.noun_chunks:
            phrase = " ".join(
                t.text for t in chunk if not t.is_stop and t.is_alpha
            ).strip().lower()
            if phrase and phrase not in _GENERIC_WORDS and phrase not in out:
                out.append(phrase)
        for tok in doc:
            if tok.pos_ in {"NOUN", "PROPN", "VERB"} and not tok.is_stop and tok.is_alpha:
                lemma = (tok.lemma_ or tok.text).lower()
                if lemma not in _GENERIC_WORDS and lemma not in out:
                    out.append(lemma)
        return tuple(out[:8])


def _to_result(m: MetricOutput) -> MetricResult:
    return MetricResult(
        value=m.value, computed=m.computed, label=m.label, detail=m.detail
    )


def _question_out(q: Question | None) -> QuestionOut | None:
    if q is None:
        return None
    return QuestionOut(id=q.id, text=q.text, category=q.category)
