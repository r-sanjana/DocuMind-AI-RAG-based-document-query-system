"""
Utils – configuration, helpers, and shared utilities
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

logger = logging.getLogger("documind.utils")

# Project root: documind-ai/ (parent of backend/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    GEMINI_API_KEY: str = ""
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    MAX_FILE_SIZE_MB: int = 50
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    LOG_LEVEL: str = "INFO"

    model_config = {
        "env_file": str(ENV_FILE),
        "case_sensitive": True,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    logger.info("Using Gemini model: %s", settings.GEMINI_MODEL)
    return settings


def format_gemini_error(exc: Exception) -> str:
    """Return a short, actionable message for Gemini API failures."""
    msg = str(exc)
    lower = msg.lower()
    if "404" in msg and "not found" in lower:
        return (
            "Gemini model not found. Set GEMINI_MODEL in .env to a model your API key supports, "
            "e.g. gemini-2.5-flash-lite or gemini-2.0-flash-lite. "
            "Then restart the backend."
        )
    if "429" in msg or "quota" in lower or "rate limit" in lower or "rate-limit" in lower:
        return (
            "Gemini API quota exceeded (free tier limit reached). "
            "Wait about 1 minute and try again, or try again tomorrow for daily limits. "
            "Try GEMINI_MODEL=gemini-2.5-flash-lite in .env (lighter quota), "
            "or enable billing at https://aistudio.google.com. "
            "Usage dashboard: https://ai.dev/rate-limit"
        )
    if "api key" in lower or "invalid" in lower and "key" in lower:
        return "Invalid or missing Gemini API key. Check GEMINI_API_KEY in your .env file."
    return f"Gemini API error: {msg}"


def configure_logging():
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_pdf(filename: str, content: bytes) -> Optional[str]:
    """
    Validate a PDF file.
    Returns an error message string or None if valid.
    """
    settings = get_settings()

    if not filename.lower().endswith(".pdf"):
        return "Only PDF files are accepted."

    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        return f"File size {size_mb:.1f}MB exceeds limit of {settings.MAX_FILE_SIZE_MB}MB."

    # Check PDF magic bytes
    if not content.startswith(b"%PDF"):
        return "File does not appear to be a valid PDF."

    return None
