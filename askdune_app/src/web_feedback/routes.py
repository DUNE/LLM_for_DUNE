
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from config import WEB_FEEDBACK_ENABLED, ENABLE_AUTHENTICATION
from src.auth.fermilab_auth import fermilab_auth
from src.web_feedback.registry import get_response_by_token
from src.web_feedback.storage import save_feedback

router = APIRouter()

ALLOWED_REASONS = {
    "Inaccurate response",
    "Incorrect or irrelevant references",
    "Outdated information",
    "Incomplete response",
    "Answer did not address my question",
    "Other",
}


class FeedbackRequest(BaseModel):
    response_id: str = Field(min_length=1, max_length=128)
    rating: str
    reasons: List[str] = Field(default_factory=list)
    comment: Optional[str] = Field(default=None, max_length=2000)


@router.post("/api/web-feedback")
async def submit_web_feedback(
    payload: FeedbackRequest,
    request: Request,
):
    if not WEB_FEEDBACK_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Web feedback is disabled",
        )


    if payload.rating not in ("positive", "negative"):
        raise HTTPException(
            status_code=422,
            detail="Invalid rating",
        )

    if any(reason not in ALLOWED_REASONS for reason in payload.reasons):
        raise HTTPException(
            status_code=422,
            detail="Invalid feedback reason",
        )

    if payload.rating == "positive" and (
        payload.reasons or payload.comment
    ):
        raise HTTPException(
            status_code=422,
            detail="Positive feedback cannot include negative reasons",
        )

    record = get_response_by_token(
        response_id=payload.response_id
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Response not found or no longer available",
        )

    save_feedback(
        response_id=payload.response_id,
        rating=payload.rating,
        reasons=payload.reasons,
        comment=payload.comment,
        question=record["question"],
        answer=record["answer"],
        references=record["references"],
    )

    return {
        "status": "success",
        "message": "Feedback saved",
    }

