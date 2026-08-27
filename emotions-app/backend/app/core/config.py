"""Application configuration, sourced from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py
_PKG_ROOT = Path(__file__).resolve().parents[1]  # emotions-app/backend/app
_BACKEND_ROOT = _PKG_ROOT.parent  # emotions-app/backend
_REPO_ROOT = _BACKEND_ROOT.parents[1]  # <repo root>

JudgeProvider = Literal["auto", "claude", "openai", "none"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT / ".env", _BACKEND_ROOT / ".env"),
        env_prefix="EVAL_",
        extra="ignore",
    )

    # --- HTTP ---
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"]
    )

    # --- data ---
    # The canonical corpus ships with the app; override with EVAL_CORPUS_PATH to
    # point at a machine-local or volume-mounted copy.
    corpus_path: Path = _PKG_ROOT / "data" / "corpus.json"
    model_cache_dir: Path = _REPO_ROOT / "model_cache"

    # --- model ids ---
    bi_encoder_model: str = "sentence-transformers/all-mpnet-base-v2"
    emotion_model: str = "SamLowe/roberta-base-go_emotions"
    spacy_model: str = "en_core_web_md"
    whisper_model: str = "base.en"
    whisper_compute_type: str = "int8"

    # --- scoring knobs ---
    # Above this cosine, we treat the corpus question as a match and use its
    # curated keywords. Below, we still score, but generically.
    question_match_threshold: float = 0.60
    # A keyword counts as "covered" when its combined lexical/semantic match
    # score (0-1) clears this.
    keyword_hit_threshold: float = 0.5

    # --- feature flags ---
    enable_emotion: bool = True
    warmup_on_startup: bool = True

    # --- optional LLM judge ---
    judge_provider: JudgeProvider = "auto"
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    # Haiku by default - the judge runs on every evaluation and only writes prose
    # (it never scores), so a fast, cheap model is the right call. Override with
    # EVAL_JUDGE_MODEL_CLAUDE for higher-quality feedback.
    judge_model_claude: str = "claude-haiku-4-5"
    judge_model_openai: str = "gpt-4o-mini"
    judge_timeout_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
