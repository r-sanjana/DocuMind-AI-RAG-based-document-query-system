"""
VectorStore Service – FAISS-based embedding storage and retrieval
Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings
"""

import logging
import os
import pickle
import shutil
from typing import List, Dict, Any, Optional

logger = logging.getLogger("documind.vectorstore")

VECTORSTORE_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")
META_PATH = os.path.join(VECTORSTORE_DIR, "metadata.pkl")


class VectorStoreService:
    """
    Manages FAISS vector index for document chunk storage and retrieval.
    Embeddings: sentence-transformers/all-MiniLM-L6-v2
    """

    def __init__(self):
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        self._index = None
        self._texts: List[str] = []
        self._metadata: List[Dict] = []
        self._embedder = None
        self._load()

    def _get_embedder(self):
        """Lazy-load the sentence transformer model"""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded SentenceTransformer: all-MiniLM-L6-v2")
        return self._embedder

    def _load(self):
        """Load existing FAISS index from disk"""
        try:
            import faiss
            if os.path.exists(INDEX_PATH + ".index") and os.path.exists(META_PATH):
                self._index = faiss.read_index(INDEX_PATH + ".index")
                with open(META_PATH, "rb") as f:
                    data = pickle.load(f)
                    self._texts = data.get("texts", [])
                    self._metadata = data.get("metadata", [])
                logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
            self._index = None

    def _save(self):
        """Persist FAISS index to disk"""
        import faiss
        faiss.write_index(self._index, INDEX_PATH + ".index")
        with open(META_PATH, "wb") as f:
            pickle.dump({"texts": self._texts, "metadata": self._metadata}, f)

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Embed chunks and add to FAISS index.
        chunks: list of {"text": ..., "metadata": {...}}
        """
        import faiss
        import numpy as np

        embedder = self._get_embedder()
        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]

        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=32)
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]

        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)  # Inner-product (cosine after normalization)
            logger.info(f"Created new FAISS IndexFlatIP, dim={dim}")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._texts.extend(texts)
        self._metadata.extend(metas)
        self._save()
        logger.info(f"Index now has {self._index.ntotal} vectors total")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar chunks for a query.
        Returns list of {"text", "metadata", "score"}
        """
        import faiss
        import numpy as np

        if self._index is None or self._index.ntotal == 0:
            raise ValueError("No documents indexed. Please upload PDFs first.")

        embedder = self._get_embedder()
        query_vec = embedder.encode([query], show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, min(top_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "text": self._texts[idx],
                "metadata": self._metadata[idx],
                "score": float(score),
            })

        return results

    def stats(self) -> Dict[str, Any]:
        """Return stats about the current index"""
        return {
            "total_vectors": self._index.ntotal if self._index else 0,
            "total_chunks": len(self._texts),
            "unique_sources": list(set(m.get("source", "") for m in self._metadata)),
        }

    def clear(self):
        """Wipe the index"""
        self._index = None
        self._texts = []
        self._metadata = []
        if os.path.exists(VECTORSTORE_DIR):
            shutil.rmtree(VECTORSTORE_DIR)
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        logger.info("Vector store cleared.")
