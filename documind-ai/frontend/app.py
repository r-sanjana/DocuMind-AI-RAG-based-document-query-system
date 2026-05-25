"""
DocuMind AI – Streamlit Frontend
Premium ChatGPT-style interface for RAG document querying
"""

import json
import os
import time
import uuid
from datetime import datetime
from io import BytesIO
from typing import Optional

import requests
import streamlit as st
import streamlit.components.v1 as components

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api/v1")

# ── Load custom CSS + ensure sidebar is expanded (Streamlit 1.57+ persists collapse in localStorage)
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles", "main.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    components.html(
        """
        <script>
        (function () {
            var storage = window.parent ? window.parent.localStorage : localStorage;
            Object.keys(storage).forEach(function (key) {
                if (key.indexOf("stSidebarCollapsed-") === 0) {
                    storage.setItem(key, "false");
                }
            });
        })();
        </script>
        """,
        height=0,
    )

load_css()

# ── Session state initialisation ─────────────────────────────────────────────
def init_session():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "uploaded_files": [],
        "dark_mode": True,
        "docs_indexed": False,
        "current_tab": "chat",
        "summary_result": None,
        "thinking": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── API helpers ───────────────────────────────────────────────────────────────
def api_upload(files) -> dict:
    file_tuples = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
    resp = requests.post(f"{BACKEND_URL}/upload", files=file_tuples, timeout=120)
    resp.raise_for_status()
    return resp.json()

def api_query(question: str, top_k: int = 2) -> dict:
    payload = {"question": question, "top_k": top_k, "session_id": st.session_state.session_id}
    resp = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()

def api_summarize(style: str = "concise") -> dict:
    payload = {"style": style}
    resp = requests.post(f"{BACKEND_URL}/summarize", json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()

def api_vectordb_stats() -> dict:
    resp = requests.get(f"{BACKEND_URL}/vectordb/stats", timeout=10)
    resp.raise_for_status()
    return resp.json()

def api_clear_docs() -> dict:
    resp = requests.delete(f"{BACKEND_URL}/documents", timeout=30)
    resp.raise_for_status()
    return resp.json()

def check_backend() -> bool:
    try:
        resp = requests.get(f"{BACKEND_URL.replace('/api/v1', '')}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo + title
        st.markdown("""
        <div class="sidebar-logo">
            <span class="logo-icon">🧠</span>
            <div>
                <div class="logo-title">DocuMind AI</div>
                <div class="logo-sub">RAG Document Intelligence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Backend status
        backend_ok = check_backend()
        status_class = "status-online" if backend_ok else "status-offline"
        status_dot = "#34d399" if backend_ok else "#f87171"
        status_text = "Backend Online" if backend_ok else "Backend Offline"
        st.markdown(
            f'<div class="status-badge {status_class}">'
            f'<span class="status-dot" style="background:{status_dot}"></span>{status_text}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # PDF Upload
        st.markdown('<div class="sidebar-section-title">📄 Upload Documents</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drag & drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
            label_visibility="collapsed",
        )

        if uploaded:
            if st.button("⚡ Process PDFs", use_container_width=True, type="primary"):
                with st.spinner("Extracting & indexing..."):
                    try:
                        result = api_upload(uploaded)
                        success = sum(1 for r in result["results"] if r["status"] == "success")
                        if success:
                            st.success(f"✅ {success} PDF(s) indexed!")
                            st.session_state.docs_indexed = True
                            st.session_state.uploaded_files = [r["filename"] for r in result["results"] if r["status"] == "success"]
                        for r in result["results"]:
                            if r["status"] == "error":
                                st.error(f"❌ {r['filename']}: {r.get('detail', 'error')}")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

        # Indexed files
        if st.session_state.uploaded_files:
            st.markdown('<div class="sidebar-section-title" style="margin-top:1rem">📚 Indexed Files</div>', unsafe_allow_html=True)
            for fname in st.session_state.uploaded_files:
                st.markdown(f'<div class="file-chip">📄 {fname}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Stats
        if st.session_state.docs_indexed:
            try:
                stats = api_vectordb_stats()
                col1, col2 = st.columns(2)
                col1.metric("Vectors", stats.get("total_vectors", 0))
                col2.metric("Chunks", stats.get("total_chunks", 0))
            except Exception:
                pass

        st.markdown("---")

        # Controls
        st.markdown('<div class="sidebar-section-title">⚙️ Controls</div>', unsafe_allow_html=True)

        top_k = st.slider("Retrieval depth (top-k)", 3, 15, 5, help="Number of chunks to retrieve")
        st.session_state.top_k = top_k

        col1, col2 = st.columns(2)
        if col1.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        if col2.button("🗂️ Clear Docs", use_container_width=True):
            try:
                api_clear_docs()
                st.session_state.uploaded_files = []
                st.session_state.docs_indexed = False
                st.success("Cleared!")
                st.rerun()
            except Exception as e:
                st.error(str(e))

        # Download chat
        if st.session_state.messages:
            chat_export = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                "💾 Export Chat",
                data=chat_export,
                file_name=f"documind_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        # Footer
        st.markdown("""
        <div class="sidebar-footer">
            <div>Built with LangChain + FAISS + Gemini</div>
            <div class="sidebar-footer-version">v1.0.0</div>
        </div>
        """, unsafe_allow_html=True)


# ── Chat rendering ────────────────────────────────────────────────────────────
def render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    if role == "user":
        st.markdown(f"""
        <div class="message-row user-row">
            <div class="message-bubble user-bubble">
                <div class="message-content">{content}</div>
            </div>
            <div class="avatar user-avatar">👤</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message-row assistant-row">
            <div class="avatar assistant-avatar">🧠</div>
            <div class="message-bubble assistant-bubble">
                <div class="message-content">{content}</div>
                {render_sources_html(sources)}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_sources_html(sources: list) -> str:
    if not sources:
        return ""
    items = "".join(
        f'<div class="source-chip">📄 {s["document"]} · pg. {s["page"]} · {s["relevance_score"]:.2f}</div>'
        for s in sources[:4]
    )
    return f'<div class="sources-row">{items}</div>'


def typing_effect(placeholder, text: str, delay: float = 0.008):
    """Stream text character by character"""
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(f"""
        <div class="message-row assistant-row">
            <div class="avatar assistant-avatar">🧠</div>
            <div class="message-bubble assistant-bubble">
                <div class="message-content">{displayed}▌</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(delay)
    # Final without cursor
    placeholder.markdown(f"""
    <div class="message-row assistant-row">
        <div class="avatar assistant-avatar">🧠</div>
        <div class="message-bubble assistant-bubble">
            <div class="message-content">{displayed}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Welcome screen ────────────────────────────────────────────────────────────
def render_welcome():
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">🧠</div>
        <h1 class="welcome-title">DocuMind AI</h1>
        <p class="welcome-subtitle">Upload your PDFs and start asking intelligent questions.<br>Powered by RAG · FAISS · Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="examples-title">Try asking</div>',
        unsafe_allow_html=True,
    )

    prompts = [
        "📋 Summarize the key points of this document",
        "🔍 What are the main conclusions?",
        "📊 Find all statistics mentioned",
        "⚖️ Compare sections A and B",
        "❓ What does the document say about [topic]?",
        "📌 List all recommendations",
    ]

    st.markdown('<div class="examples-grid">', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, prompt in enumerate(prompts):
        with cols[i % 3]:
            if st.button(prompt, use_container_width=True, key=f"prompt_{i}", type="secondary"):
                clean = prompt.split(" ", 1)[1]  # strip emoji
                handle_query(clean)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Query handler ─────────────────────────────────────────────────────────────
def handle_query(question: str):
    if not question.strip():
        return

    st.session_state.messages.append({"role": "user", "content": question, "sources": []})

    with st.spinner("🔍 Searching documents..."):
        try:
            result = api_query(question, top_k=st.session_state.get("top_k", 5))
            answer = result.get("answer", "No answer returned.")
            sources = result.get("sources", [])
        except requests.exceptions.ConnectionError:
            answer = "❌ Cannot connect to backend. Please ensure the FastAPI server is running."
            sources = []
        except Exception as e:
            answer = f"❌ Error: {str(e)}"
            sources = []

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()


# ── Summary tab ───────────────────────────────────────────────────────────────
def render_summary_tab():
    st.markdown('<h2 class="tab-title">📋 Document Summarizer</h2>', unsafe_allow_html=True)

    if not st.session_state.docs_indexed:
        st.info("📤 Upload and process PDFs first to generate summaries.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        style = st.selectbox("Summary style", ["concise", "detailed", "bullet"], index=0)
    with col2:
        st.write("")
        st.write("")
        generate = st.button("✨ Generate Summary", type="primary", use_container_width=True)

    if generate:
        with st.spinner("Generating summary with Gemini..."):
            try:
                result = api_summarize(style)
                st.session_state.summary_result = result
            except Exception as e:
                st.error(f"Summarization failed: {e}")

    if st.session_state.summary_result:
        res = st.session_state.summary_result

        st.markdown("### 📝 Summary")
        st.markdown(f'<div class="summary-card">{res.get("summary", "")}</div>', unsafe_allow_html=True)

        if res.get("key_points"):
            st.markdown("### 🔑 Key Points")
            for kp in res["key_points"]:
                st.markdown(f'<div class="key-point-chip">• {kp}</div>', unsafe_allow_html=True)

        st.download_button(
            "💾 Download Summary",
            data=json.dumps(res, indent=2),
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


# ── About tab ─────────────────────────────────────────────────────────────────
def render_about_tab():
    # HTML must start at column 0 — leading spaces make Streamlit treat the rest as a code block.
    st.markdown(
        """<div class="about-container">
<h2>🧠 About DocuMind AI</h2>
<p>DocuMind AI is a production-grade RAG (Retrieval Augmented Generation) system for intelligent document querying.</p>
<h3>🏗️ Architecture</h3>
<div class="arch-flow">PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Index → Similarity Search → Gemini LLM → Answer</div>
<h3>🛠️ Tech Stack</h3>
<div class="tech-grid">
<div class="tech-card">⚡ FastAPI</div>
<div class="tech-card">🎈 Streamlit</div>
<div class="tech-card">🔗 LangChain</div>
<div class="tech-card">🧮 FAISS</div>
<div class="tech-card">✨ Gemini AI</div>
<div class="tech-card">🤗 Sentence Transformers</div>
</div>
<h3>🚀 Features</h3>
<ul>
<li>Multi-PDF upload and indexing</li>
<li>Semantic vector search with FAISS</li>
<li>Source citations with page numbers</li>
<li>Conversational chat with memory</li>
<li>Document summarization (3 styles)</li>
<li>Chat history export</li>
</ul>
</div>""",
        unsafe_allow_html=True,
    )


# ── Main layout ───────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📋 Summarize", "ℹ️ About"])

    with tab1:
        # Welcome or chat
        if not st.session_state.messages:
            render_welcome()
        else:
            # Render chat history
            for msg in st.session_state.messages:
                render_message(msg)

        question = st.chat_input("Ask anything about your documents…")

        if question:
            handle_query(question)

    with tab2:
        render_summary_tab()

    with tab3:
        render_about_tab()


if __name__ == "__main__":
    main()
