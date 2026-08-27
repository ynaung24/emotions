from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_corpus
from app.schemas.evaluation import QuestionListResponse, QuestionOut
from app.services.corpus import Corpus

router = APIRouter(tags=["questions"])


@router.get("/questions", response_model=QuestionListResponse)
def list_questions(corpus: Corpus = Depends(get_corpus)) -> QuestionListResponse:
    return QuestionListResponse(
        questions=[
            QuestionOut(id=q.id, text=q.text, category=q.category) for q in corpus.questions
        ]
    )
