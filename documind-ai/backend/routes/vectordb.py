"""
VectorDB router – vector store management
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.services.vectorstore_service import VectorStoreService

logger = logging.getLogger("documind.vectordb")
router = APIRouter()


@router.get("/vectordb/stats")
async def vectordb_stats():
    """Return statistics about the current vector store"""
    try:
        svc = VectorStoreService()
        stats = svc.stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vectordb")
async def clear_vectordb():
    """Clear the entire vector store"""
    try:
        svc = VectorStoreService()
        svc.clear()
        return {"message": "Vector store cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
