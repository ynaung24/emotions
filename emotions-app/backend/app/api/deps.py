from __future__ import annotations

from fastapi import Request

from app.services.corpus import Corpus
from app.services.evaluator import Evaluator
from app.services.registry import ModelRegistry


def get_evaluator(request: Request) -> Evaluator:
    evaluator: Evaluator = request.app.state.evaluator
    return evaluator


def get_registry(request: Request) -> ModelRegistry:
    registry: ModelRegistry = request.app.state.registry
    return registry


def get_corpus(request: Request) -> Corpus:
    return get_registry(request).corpus
