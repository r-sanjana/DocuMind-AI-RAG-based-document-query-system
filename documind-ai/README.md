# 🧠 DocuMind AI

> **RAG-Based Document Intelligence System** — Upload PDFs, ask questions, get cited answers powered by Gemini AI

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-ff4b4b?logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1c3c3c)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-4285f4?logo=google)](https://ai.google.dev)
[![FAISS](https://img.shields.io/badge/FAISS-1.8-0078d7)](https://github.com/facebookresearch/faiss)

---

## 📸 Preview

```
╔══════════════════════════════════════════════════════════════╗
║  🧠 DocuMind AI          💬 Chat  📋 Summarize  ℹ️ About   ║
╠════════════════╦═════════════════════════════════════════════╣
║  📄 Upload     ║                                             ║
║  ┌──────────┐  ║   🧠 DocuMind AI                           ║
║  │ Drop PDF │  ║   Upload your PDFs and start asking        ║
║  └──────────┘  ║   intelligent questions.                   ║
║                ║                                             ║
║  📚 Files      ║   ✨ Try asking:                            ║
║  ▸ doc1.pdf    ║   [Summarize key points] [Find statistics] ║
║  ▸ doc2.pdf    ║                                             ║
║                ║   👤 What are the main findings?           ║
║  Vectors: 248  ║                                             ║
║  Chunks:  248  ║   🧠 Based on the document, the main       ║
║                ║   findings are... [📄 doc1.pdf · pg. 3]    ║
║  ⚙️ Controls   ║                                             ║
║  top-k: [5──] ║   ┌─────────────────────────────────────┐  ║
║  [Clear Chat]  ║   │ Ask anything about your documents…  │  ║
║  [Export Chat] ║   └─────────────────────────────────────┘  ║
╚════════════════╩═════════════════════════════════════════════╝
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 📤 **Multi-PDF Upload** | Upload & process multiple PDFs simultaneously |
| 🔍 **Semantic Search** | FAISS vector similarity search across all documents |
| 🤖 **RAG Pipeline** | Context-aware answers grounded in your documents |
| 📌 **Source Citations** | Every answer includes document name + page number |
| 💬 **Chat Memory** | Conversational context preserved within sessions |
| 📋 **Smart Summarization** | Concise / Detailed / Bullet-point summary styles |
| 📥 **Export History** | Download full chat as JSON |
| 🔒 **Hallucination Guard** | Gemini restricted to retrieved context only |
| 🖥️ **Premium Dark UI** | ChatGPT-style interface built with Streamlit |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DocuMind AI                             │
│                                                                  │
│  ┌──────────┐    ┌──────────────────────────────────────────┐   │
│  │Streamlit │───▶│              FastAPI Backend              │   │
│  │Frontend  │    │                                          │   │
│  └──────────┘    │  ┌──────────┐  ┌─────────┐  ┌────────┐  │   │
│                  │  │  /upload │  │ /query  │  │/summar │  │   │
│  PDF Upload ────▶│  └────┬─────┘  └────┬────┘  └────┬───┘  │   │
│                  │       │             │             │       │   │
│                  │  ┌────▼─────────────▼─────────────▼────┐  │   │
│                  │  │           PDF Service               │  │   │
│                  │  │  Extract → Chunk → Embed            │  │   │
│                  │  └──────────────────┬──────────────────┘  │   │
│                  │                     │                      │   │
│                  │  ┌──────────────────▼──────────────────┐  │   │
│                  │  │         VectorStore Service          │  │   │
│                  │  │    FAISS (all-MiniLM-L6-v2)         │  │   │
│                  │  └──────────────────┬──────────────────┘  │   │
│                  │                     │ Semantic Search      │   │
│                  │  ┌──────────────────▼──────────────────┐  │   │
│                  │  │            RAG Service               │  │   │
│                  │  │  Context + Prompt → Gemini 1.5 Flash│  │   │
│                  │  └──────────────────┬──────────────────┘  │   │
│                  │                     │ Answer + Citations   │   │
│                  └─────────────────────┼────────────────────┘   │
│                                        ▼                         │
│                              Streamlit Chat UI                   │
└─────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline Flow
```
PDF → Text Extraction → Recursive Text Splitting → Sentence Transformer Embeddings
    → FAISS IndexFlatIP → Cosine Similarity Search → Top-K Chunks
    → Context-augmented Gemini Prompt → Grounded Answer + Source Citations
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit 1.35, Custom CSS |
| **Backend** | FastAPI 0.111, Uvicorn |
| **LLM** | Google Gemini 1.5 Flash |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector DB** | FAISS (Facebook AI Similarity Search) |
| **RAG Framework** | LangChain 0.2 |
| **PDF Parsing** | pdfplumber, PyPDF2 |
| **OCR** | pytesseract (Tesseract fallback) |
| **Config** | Pydantic Settings, python-dotenv |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Gemini API Key](https://aistudio.google.com/app/apikey) (free tier available)
- Tesseract OCR (optional, for scanned PDFs)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/documind-ai.git
cd documind-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 5. Start the backend
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Start the frontend (new terminal)
```bash
cd frontend
streamlit run app.py
```

### 7. Open in browser
```
Frontend:  http://localhost:8501
API Docs:  http://localhost:8000/docs
```

---

## 📖 Usage Guide

### Basic Workflow
1. **Upload PDFs** — Use the sidebar to drag and drop one or more PDFs
2. **Process** — Click "⚡ Process PDFs" to extract, chunk, and index
3. **Ask** — Type a question in the chat input or click an example prompt
4. **Review** — Read the answer and check the source citations below each response
5. **Summarize** — Switch to the Summarize tab for document-level summaries

### Example Prompts
- *"What are the main conclusions of this research?"*
- *"Find all dates and timelines mentioned"*
- *"Summarize the methodology section"*
- *"What risks are identified in the document?"*
- *"Compare the findings in chapters 2 and 3"*

### API Endpoints
```
POST   /api/v1/upload           Upload PDFs
POST   /api/v1/query            Query documents (RAG)
POST   /api/v1/summarize        Summarize documents
GET    /api/v1/documents        List indexed documents
DELETE /api/v1/documents        Clear all documents
POST   /api/v1/chat/session     Create chat session
GET    /api/v1/chat/session/:id Get chat history
GET    /api/v1/vectordb/stats   Vector store statistics
```

---

## 🚢 Deployment

### Render (Recommended)
```bash
# Deploy using render.yaml
# 1. Push to GitHub
# 2. Connect repo to Render
# 3. Render auto-detects render.yaml
# 4. Set GEMINI_API_KEY in environment variables
```

### Streamlit Cloud (Frontend only)
```
1. Push to GitHub
2. Go to share.streamlit.io
3. Deploy from frontend/ directory
4. Set BACKEND_URL in Streamlit secrets
```

### Docker
```bash
# Build
docker build -t documind-backend ./backend
docker build -t documind-frontend ./frontend

# Run
docker run -p 8000:8000 -e GEMINI_API_KEY=... documind-backend
docker run -p 8501:8501 -e BACKEND_URL=http://... documind-frontend
```

---

## 📁 Project Structure

```
documind-ai/
│
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── routes/
│   │   ├── upload.py              # PDF upload & processing
│   │   ├── query.py               # RAG query endpoint
│   │   ├── summarize.py           # Summarization endpoint
│   │   ├── chat.py                # Chat session management
│   │   └── vectordb.py            # Vector store management
│   ├── services/
│   │   ├── pdf_service.py         # PDF extraction & chunking
│   │   ├── vectorstore_service.py # FAISS operations
│   │   ├── rag_service.py         # RAG pipeline + Gemini
│   │   └── summarize_service.py   # Summarization logic
│   ├── utils/
│   │   └── config.py              # Settings & helpers
│   ├── vectorstore/               # FAISS index storage
│   └── requirements.txt
│
├── frontend/
│   ├── app.py                     # Main Streamlit app
│   ├── styles/
│   │   └── main.css               # Premium dark theme CSS
│   ├── .streamlit/
│   │   └── config.toml            # Streamlit config
│   └── requirements.txt
│
├── data/
│   └── uploads/                   # Uploaded PDF storage
│
├── render.yaml                    # Render deployment config
├── requirements.txt               # Combined dependencies
├── .env.example                   # Environment template
└── README.md
```

---

## 🔮 Future Enhancements

- [ ] **Voice Input** — Web Speech API integration
- [ ] **Multi-user Auth** — JWT authentication + user-scoped vector stores
- [ ] **Redis Sessions** — Persistent chat sessions across restarts
- [ ] **Streaming Responses** — Server-sent events for real-time typing
- [ ] **Document Comparison** — Side-by-side analysis of multiple PDFs
- [ ] **Knowledge Graph** — Entity extraction and relationship mapping
- [ ] **Fine-tuned Embeddings** — Domain-specific embedding models
- [ ] **Webhook Support** — Auto-ingest documents from URLs/cloud storage
- [ ] **Analytics Dashboard** — Query patterns, popular topics, usage stats

---

## 🤝 Contributing

```bash
git checkout -b feature/your-feature
# Make changes
git commit -m "feat: add your feature"
git push origin feature/your-feature
# Open a Pull Request
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Built with ❤️ using LangChain · FAISS · Gemini · FastAPI · Streamlit
</div>
