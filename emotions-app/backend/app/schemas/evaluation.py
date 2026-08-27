"""Request/response models for the evaluation API (pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QuestionOut(BaseModel):
    id: str
    text: str
    category: str


class QuestionListResponse(BaseModel):
    questions: list[QuestionOut]


class TextEvaluationRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=20_000)
    role: str | None = Field(default=None, max_length=120)


class MetricResult(BaseModel):
    """One scored dimension.

    `computed=False` means we could not measure this dimension for this input
    (e.g. delivery metrics on a text answer). The UI must not render a value
    when `computed` is false - that was the old bug where a hardcoded 75 showed
    up as if it were real.
    """

    value: float = Field(ge=0, le=100)
    computed: bool = True
    label: str
    detail: str | None = None


class EmotionScore(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)


class DeliveryMetrics(BaseModel):
    words_per_minute: float
    pause_ratio: float = Field(ge=0, le=1)
    filler_rate_per_100: float
    pitch_variation: float
    duration_seconds: float


class FeedbackBlock(BaseModel):
    summary: str
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    source: str = "template"  # "template" | "claude" | "openai"


class EvaluationResult(BaseModel):
    score: float = Field(ge=0, le=100)
    metrics: dict[str, MetricResult]
    feedback: FeedbackBlock
    matched_question: QuestionOut | None = None
    scored_generically: bool = False
    emotions: list[EmotionScore] = Field(default_factory=list)
    transcript: str | None = None
    delivery: DeliveryMetrics | None = None
    flags: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    models: dict[str, bool]
