"""
PDF Service – text extraction and chunking
Supports both digital PDFs (pdfplumber) and OCR fallback (pytesseract)
"""

import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger("documind.pdf_service")


class PDFService:
    """Handles PDF text extraction and document chunking"""

    CHUNK_SIZE = 1000      # characters per chunk
    CHUNK_OVERLAP = 200    # overlap between chunks

    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from a PDF file, page by page.
        Falls back to OCR for scanned pages.

        Returns:
            List of dicts with keys: page_number, text, char_count
        """
        pages = []

        # Primary: pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append({
                            "page_number": i,
                            "text": text.strip(),
                            "char_count": len(text.strip()),
                        })
                    else:
                        # Try OCR fallback for image-based pages
                        ocr_text = self._ocr_page(page)
                        if ocr_text:
                            pages.append({
                                "page_number": i,
                                "text": ocr_text,
                                "char_count": len(ocr_text),
                                "ocr": True,
                            })
        except ImportError:
            # Fallback: PyPDF2
            try:
                import PyPDF2
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for i, page in enumerate(reader.pages, start=1):
                        text = page.extract_text() or ""
                        if text.strip():
                            pages.append({
                                "page_number": i,
                                "text": text.strip(),
                                "char_count": len(text.strip()),
                            })
            except Exception as e:
                logger.error(f"PyPDF2 extraction failed: {e}")
                raise
        except Exception as e:
            logger.error(f"PDF extraction error for {file_path}: {e}")
            raise

        logger.info(f"Extracted {len(pages)} pages from {os.path.basename(file_path)}")
        return pages

    def _ocr_page(self, page) -> str:
        """OCR a single pdfplumber page using pytesseract"""
        try:
            import pytesseract
            from PIL import Image
            img = page.to_image(resolution=200).original
            return pytesseract.image_to_string(img)
        except ImportError:
            return ""
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""

    def chunk_text(
        self,
        pages: List[Dict[str, Any]],
        source_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Split pages into overlapping chunks for embedding.

        Returns:
            List of chunk dicts with: text, metadata (source, page, chunk_id)
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = []
        chunk_id = 0

        for page_data in pages:
            page_chunks = splitter.split_text(page_data["text"])
            for chunk_text in page_chunks:
                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text.strip(),
                        "metadata": {
                            "source": source_name,
                            "page": page_data["page_number"],
                            "chunk_id": chunk_id,
                        },
                    })
                    chunk_id += 1

        logger.info(f"Created {len(chunks)} chunks from '{source_name}'")
        return chunks
