"""
Summarize Service – generates document summaries using Gemini
"""

import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger("documind.summarize")

SUMMARIZE_PROMPTS = {
    "concise": "Provide a concise executive summary (3-5 sentences) of the following document content. Highlight the most important points.",
    "detailed": "Provide a comprehensive and detailed summary of the following document content. Cover all major topics, key arguments, and important details.",
    "bullet": "Summarize the following document content as a structured list of key points and takeaways. Use clear, concise bullet points organized by topic.",
}


class SummarizeService:
    """Generates AI-powered document summaries"""

    def __init__(self):
        from backend.utils.config import get_settings

        settings = get_settings()
        self.gemini_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = settings.GEMINI_MODEL

    def summarize(self, filename: Optional[str] = None, style: str = "concise") -> Dict[str, Any]:
        """
        Generate a summary of one or all uploaded documents.

        Args:
            filename: specific doc to summarize, or None for all
            style: "concise" | "detailed" | "bullet"
        """
        from backend.services.vectorstore_service import VectorStoreService

        vs = VectorStoreService()
        stats = vs.stats()

        if stats["total_vectors"] == 0:
            raise ValueError("No documents indexed. Please upload PDFs first.")

        # Collect text: retrieve a broad sample with a neutral query
        chunks = vs.search("main topics overview summary key points", top_k=15)

        if filename:
            chunks = [c for c in chunks if c["metadata"].get("source", "") == filename]
            if not chunks:
                raise ValueError(f"No indexed content found for '{filename}'.")

        combined_text = "\n\n".join(c["text"] for c in chunks)
        prompt_instruction = SUMMARIZE_PROMPTS.get(style, SUMMARIZE_PROMPTS["concise"])

        summary = self._call_gemini(prompt_instruction, combined_text)
        key_points = self._extract_key_points(combined_text)

        return {
            "summary": summary,
            "key_points": key_points,
            "style": style,
            "documents_included": list(set(c["metadata"].get("source", "") for c in chunks)),
            "chunks_analyzed": len(chunks),
        }

    def _call_gemini(self, instruction: str, content: str) -> str:
        """Call Gemini for summarization"""
        try:
            import google.generativeai as genai
            from backend.utils.config import format_gemini_error

            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel(
                self.gemini_model,
                generation_config={"temperature": 0.3, "max_output_tokens": 2048},
            )
            prompt = f"{instruction}\n\nDocument Content:\n{content}"
            response = model.generate_content(prompt)
            return response.text

        except ImportError:
            return f"[Summary] {content[:500]}..."
        except Exception as e:
            logger.error(f"Gemini summarize error: {e}")
            raise RuntimeError(format_gemini_error(e)) from e

    def _extract_key_points(self, content: str) -> list:
        """Extract key points using Gemini"""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel(
                self.gemini_model,
                generation_config={"temperature": 0.2, "max_output_tokens": 512},
            )
            prompt = (
                "Extract exactly 5 key points from the following text. "
                "Return ONLY a JSON array of strings, no preamble.\n\n"
                f"Text:\n{content[:3000]}"
            )
            response = model.generate_content(prompt)
            import json, re
            raw = response.text
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return [line.strip("- ").strip() for line in raw.split("\n") if line.strip()][:5]
        except Exception as e:
            logger.warning(f"Key point extraction failed: {e}")
            return []
