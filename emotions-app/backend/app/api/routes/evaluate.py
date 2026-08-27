from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_evaluator
from app.core.logging import get_logger
from app.schemas.evaluation import EvaluationResult, TextEvaluationRequest
from app.services.evaluator import Evaluator

logger = get_logger(__name__)
router = APIRouter(prefix="/evaluate", tags=["evaluate"])

_MAX_AUDIO_BYTES = 25 * 1024 * 1024


@router.post("/text", response_model=EvaluationResult)
def evaluate_text(
    payload: TextEvaluationRequest,
    evaluator: Evaluator = Depends(get_evaluator),
) -> EvaluationResult:
    # sync def -> FastAPI runs this in a worker thread, so the CPU-bound scoring
    # never blocks the event loop.
    logger.info("evaluate/text q=%r len(answer)=%d", payload.question[:60], len(payload.answer))
    return evaluator.evaluate_text(payload.question, payload.answer, payload.role)


@router.post("/audio", response_model=EvaluationResult)
async def evaluate_audio(
    file: UploadFile = File(...),
    question: str = Form(..., min_length=1, max_length=1000),
    evaluator: Evaluator = Depends(get_evaluator),
) -> EvaluationResult:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    if len(raw) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (25 MB max).")

    logger.info("evaluate/audio q=%r bytes=%d", question[:60], len(raw))
    try:
        return await run_in_threadpool(evaluator.evaluate_audio, question, raw)
    except Exception as exc:
        logger.exception("audio evaluation failed")
        raise HTTPException(
            status_code=422,
            detail=f"Could not process the audio: {exc}",
        ) from exc
