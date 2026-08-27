from __future__ import annotations

from app.core.config import Settings
from app.services.feedback.judge import NullJudge, _parse, build_judge


def test_build_judge_returns_null_without_keys():
    s = Settings(judge_provider="auto", ANTHROPIC_API_KEY=None, OPENAI_API_KEY=None)
    assert isinstance(build_judge(s), NullJudge)


def test_build_judge_none_provider_short_circuits():
    s = Settings(judge_provider="none", ANTHROPIC_API_KEY="x")
    assert isinstance(build_judge(s), NullJudge)


def test_parse_handles_fenced_json():
    fb = _parse('```json\n{"summary": "ok", "strengths": ["a"], "improvements": []}\n```', "claude")
    assert fb is not None
    assert fb.summary == "ok"
    assert fb.source == "claude"


def test_parse_rejects_garbage():
    assert _parse("not json at all", "openai") is None
