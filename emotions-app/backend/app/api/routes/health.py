from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_registry
from app.schemas.evaluation import HealthResponse, ReadyResponse
from app.services.registry import ModelRegistry

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
def ready(registry: ModelRegistry = Depends(get_registry)) -> ReadyResponse:
    status = registry.status()
    return ReadyResponse(ready=all(status.values()), models=status)
