"""
Summarize router – PDF summarization endpoint
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.summarize_service import SummarizeService

logger = logging.getLogger("documind.summarize")
router = APIRouter()


class SummarizeRequest(BaseModel):
    filename: Optional[str] = Field(default=None, description="Specific filename to summarize; None = all docs")
    style: str = Field(default="concise", pattern="^(concise|detailed|bullet)$")


@router.post("/summarize")
async def summarize_document(request: SummarizeRequest):
    """
    Generate a summary of uploaded PDF(s).
    Styles: concise | detailed | bullet
    """
    try:
        svc = SummarizeService()
        result = svc.summarize(filename=request.filename, style=request.style)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
