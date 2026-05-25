"""
Query router – RAG pipeline for document Q&A
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.rag_service import RAGService

logger = logging.getLogger("documind.query")
router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    session_id: Optional[str] = Field(default=None, description="Session ID for chat memory")


class QueryResponse(BaseModel):
    answer: str
    sources: list
    session_id: Optional[str]
    tokens_used: Optional[int]


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query uploaded documents using RAG pipeline.
    Retrieves relevant chunks via FAISS and generates answer with Gemini.
    """
    try:
        rag = RAGService()

        result = rag.query(
        question=request.question,
        top_k=request.top_k,
        session_id=request.session_id,
        )
        return result
    except Exception as e:
        logger.exception("Query failed")
        return {
            "answer": str(e),
            "sources": [],
            "session_id": request.session_id,
            "tokens_used": 0,
        }