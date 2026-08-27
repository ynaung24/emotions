"""Optional LLM judge - adds a written narrative on top of the deterministic scores.

Provider-agnostic: a `Judge` protocol with Claude and OpenAI adapters plus a
`NullJudge`. With no API key configured the factory returns `NullJudge` and the
API still works - the judge only ever *adds* prose, never a score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.evaluation import FeedbackBlock
from app.services.scoring.base import MetricOutput

logger = get_logger(__name__)

_SYSTEM = (
    "You are an experienced interview coach. You are given a candidate's answer, the "
    "question, and deterministic rubric scores that have already been computed. Do not "
    "re-score. Write concise, specific, actionable feedback that a candidate can act on. "
    "Respond ONLY with minified JSON of the form "
    '{"summary": str, "strengths": [str, ...], "improvements": [str, ...]}. '
    "Keep summary to one or two sentences; 2-4 bullets each for strengths and improvements."
)


@dataclass
class JudgeContext:
    question: str
    answer: str
    overall_score: float
    metrics: dict[str, MetricOutput]
    transcript: str | None = None

    def to_prompt(self) -> str:
        lines = [
            f"QUESTION: {self.question}",
            f"ANSWER: {self.answer}",
            f"OVERALL SCORE: {self.overall_score:.0f}/100",
            "RUBRIC SCORES:",
        ]
        for _name, m in self.metrics.items():
            if m.computed:
                lines.append(f"  - {m.label}: {m.value:.0f}/100 ({m.detail})")
        return "\n".join(lines)


class Judge(Protocol):
    name: str

    def generate(self, ctx: JudgeContext) -> FeedbackBlock | None: ...


class NullJudge:
    name = "none"

    def generate(self, ctx: JudgeContext) -> FeedbackBlock | None:
        return None


def _parse(raw: str, source: str) -> FeedbackBlock | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        data = json.loads(raw)
        return FeedbackBlock(
            summary=str(data["summary"]).strip(),
            strengths=[str(s).strip() for s in data.get("strengths", [])][:4],
            improvements=[str(s).strip() for s in data.get("improvements", [])][:4],
            source=source,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("judge (%s) returned unparseable output: %s", source, exc)
        return None


class ClaudeJudge:
    name = "claude"

    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self._model = model

    def generate(self, ctx: JudgeContext) -> FeedbackBlock | None:
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM,
                messages=[{"role": "user", "content": ctx.to_prompt()}],
            )
        except Exception as exc:  # network/rate-limit/etc - degrade, never fail the request
            logger.warning("Claude judge call failed: %s", exc)
            return None
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
        )
        return _parse(text, self.name)


class OpenAIJudge:
    name = "openai"

    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        import openai

        self._client = openai.OpenAI(api_key=api_key, timeout=timeout)
        self._model = model

    def generate(self, ctx: JudgeContext) -> FeedbackBlock | None:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": ctx.to_prompt()},
                ],
            )
        except Exception as exc:
            logger.warning("OpenAI judge call failed: %s", exc)
            return None
        return _parse(resp.choices[0].message.content or "", self.name)


def build_judge(settings: Settings) -> Judge:
    provider = settings.judge_provider
    if provider == "none":
        return NullJudge()

    if provider in ("auto", "claude") and settings.anthropic_api_key:
        try:
            claude: Judge = ClaudeJudge(
                settings.anthropic_api_key,
                settings.judge_model_claude,
                settings.judge_timeout_seconds,
            )
            logger.info("LLM judge: Claude (%s)", settings.judge_model_claude)
            return claude
        except ImportError:
            logger.warning("judge_provider wants Claude but `anthropic` is not installed")

    if provider in ("auto", "openai") and settings.openai_api_key:
        try:
            gpt: Judge = OpenAIJudge(
                settings.openai_api_key,
                settings.judge_model_openai,
                settings.judge_timeout_seconds,
            )
            logger.info("LLM judge: OpenAI (%s)", settings.judge_model_openai)
            return gpt
        except ImportError:
            logger.warning("judge_provider wants OpenAI but `openai` is not installed")

    logger.info("LLM judge: none (no API key configured) - feedback will be template-only")
    return NullJudge()
