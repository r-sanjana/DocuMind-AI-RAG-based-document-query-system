"""
Chat router – conversational history management
"""

import logging
from typing import List, Optional
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("documind.chat")
router = APIRouter()

# In-memory session store (replace with Redis for production)
_sessions: dict = {}


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    sources: Optional[List[dict]] = None


class ChatSession(BaseModel):
    session_id: str
    messages: List[Message]


@router.post("/chat/session")
async def create_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = []
    return {"session_id": session_id}


@router.get("/chat/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve chat history for a session"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "messages": _sessions[session_id]}


@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """Clear chat history for a session"""
    if session_id in _sessions:
        _sessions[session_id] = []
    return {"message": "Session cleared.", "session_id": session_id}


@router.get("/chat/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": [
            {"session_id": sid, "message_count": len(msgs)}
            for sid, msgs in _sessions.items()
        ]
    }


def append_to_session(session_id: str, role: str, content: str, sources: Optional[List[dict]] = None):
    """Helper used by RAG service to persist messages"""
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append({"role": role, "content": content, "sources": sources or []})
