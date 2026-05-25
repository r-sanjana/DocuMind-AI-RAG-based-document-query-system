"""
DocuMind AI - FastAPI Backend
Main application entry point
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from backend.routes import upload, query, summarize, chat, vectordb
from backend.utils.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("documind")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    settings = get_settings()
    logger.info("🚀 DocuMind AI backend starting up...")
    logger.info("Gemini model: %s", settings.GEMINI_MODEL)
    os.makedirs("vectorstore", exist_ok=True)
    os.makedirs("../data/uploads", exist_ok=True)
    yield
    logger.info("🛑 DocuMind AI backend shutting down...")


app = FastAPI(
    title="DocuMind AI",
    description="RAG-Based Document Query System powered by Gemini & FAISS",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(summarize.router, prefix="/api/v1", tags=["Summarize"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(vectordb.router, prefix="/api/v1", tags=["VectorDB"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "app": "DocuMind AI",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
