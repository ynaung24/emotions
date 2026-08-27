"""FastAPI app factory. Models load exactly once, here, during the lifespan."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.api.routes import evaluate, health, questions
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.corpus import load_corpus
from app.services.evaluator import Evaluator
from app.services.feedback.judge import build_judge
from app.services.registry import ModelRegistry

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    logger.info("loading corpus from %s", settings.corpus_path)
    corpus = load_corpus(settings.corpus_path)

    registry = ModelRegistry(settings, corpus)
    judge = build_judge(settings)

    app.state.settings = settings
    app.state.registry = registry
    app.state.evaluator = Evaluator(settings, registry, judge)

    if settings.warmup_on_startup:
        await run_in_threadpool(registry.warmup)
        logger.info("build counts: %s", registry.build_counts)

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Interview Response Evaluator",
        version=__version__,
        summary="Deterministic scoring of interview answers, with an optional LLM judge.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(questions.router)
    app.include_router(evaluate.router)
    return app


app = create_app()
