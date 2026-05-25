"""
RAG Service – Retrieval Augmented Generation pipeline
Retrieves context from FAISS → prompts Gemini → returns answer + citations
"""
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)


import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger("documind.rag")

RAG_SYSTEM_PROMPT = """You are DocuMind AI, an expert document analysis assistant.

Your task is to answer the user's question STRICTLY based on the provided document context.

Rules:
1. Only use information present in the provided context.
2. If the answer is not in the context, say: "I couldn't find relevant information in the uploaded documents."
3. Always cite the source document and page number when referencing information.
4. Be concise, accurate, and helpful.
5. Format your response clearly with proper structure when needed.
6. Do NOT hallucinate or add information not present in the context.

Context from uploaded documents:
{context}
"""


class RAGService:
    """Orchestrates the full RAG pipeline"""

    def __init__(self):
        from backend.utils.config import get_settings

        settings = get_settings()
        self.gemini_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = settings.GEMINI_MODEL
        if not self.gemini_key:
            logger.warning("GEMINI_API_KEY not set – LLM calls will fail")

    def query(
        self,
        question: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline:
        1. Retrieve relevant chunks from FAISS
        2. Build context string
        3. Query Gemini with context + question
        4. Return answer + source citations
        """
        from backend.services.vectorstore_service import VectorStoreService

        # Step 1: Retrieve
        vs = VectorStoreService()
        chunks = vs.search(question, top_k=top_k)

        if not chunks:
            return {
                "answer": "No documents have been uploaded yet. Please upload PDFs first.",
                "sources": [],
                "session_id": session_id,
                "tokens_used": 0,
            }

        # Step 2: Build context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            context_parts.append(
                f"[{i}] Source: {meta.get('source', 'Unknown')} | Page: {meta.get('page', '?')}\n{chunk['text']}"
            )
        context_str = "\n\n---\n\n".join(context_parts)

        # Step 3: Get chat history for conversational context
        history_context = ""
        if session_id:
            history_context = self._get_history_context(session_id)

        # Step 4: Call Gemini
        answer = self._call_gemini(question, context_str, history_context)

        # Step 5: Format sources
        sources = self._format_sources(chunks)

        # Step 6: Persist to session
        if session_id:
            self._save_to_session(session_id, question, answer, sources)

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "tokens_used": None,
        }

    def _call_gemini(self, question: str, context: str, history: str = "") -> str:
        """Call Google Gemini API with context-augmented prompt"""
        try:
            import google.generativeai as genai
            from backend.utils.config import format_gemini_error

            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel(
                model_name=self.gemini_model,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_output_tokens": 2048,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                ],
            )

            system_msg = RAG_SYSTEM_PROMPT.format(context=context)
            if history:
                full_prompt = f"{system_msg}\n\nConversation History:\n{history}\n\nUser Question: {question}"
            else:
                full_prompt = f"{system_msg}\n\nUser Question: {question}"

            response = model.generate_content(full_prompt)

            # Primary response extraction
            if hasattr(response, "text") and response.text:
                return response.text

            # Fallback extraction for Gemini structured responses
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content.parts:
                    return candidate.content.parts[0].text

            return "No response generated."

        except ImportError:
            return self._fallback_answer(question, context)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise RuntimeError(format_gemini_error(e)) from e

    def _fallback_answer(self, question: str, context: str) -> str:
        """Fallback when google-generativeai is not installed"""
        return (
            f"⚠️ Gemini API not configured. Here is the retrieved context for your question:\n\n"
            f"**Question:** {question}\n\n"
            f"**Retrieved Context:**\n{context[:1000]}..."
        )

    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Deduplicate and format source citations"""
        seen = set()
        sources = []
        for chunk in chunks:
            meta = chunk["metadata"]
            key = (meta.get("source", ""), meta.get("page", ""))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document": meta.get("source", "Unknown"),
                    "page": meta.get("page", "?"),
                    "relevance_score": round(chunk["score"], 4),
                    "snippet": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                })
        return sources

    def _get_history_context(self, session_id: str) -> str:
        """Retrieve last 4 message pairs for conversational context"""
        try:
            from backend.routes.chat import _sessions
            messages = _sessions.get(session_id, [])
            recent = messages[-8:]  # last 4 pairs
            return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
        except Exception:
            return ""

    def _save_to_session(self, session_id: str, question: str, answer: str, sources: list):
        """Persist Q&A to session store"""
        try:
            from backend.routes.chat import _sessions
            if session_id not in _sessions:
                _sessions[session_id] = []
            _sessions[session_id].append({"role": "user", "content": question, "sources": []})
            _sessions[session_id].append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:
            logger.warning(f"Could not save to session: {e}")
