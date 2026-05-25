"""
Upload router – handles PDF file ingestion
"""

import logging
import os
import shutil
import uuid
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.services.pdf_service import PDFService
from backend.services.vectorstore_service import VectorStoreService

logger = logging.getLogger("documind.upload")
router = APIRouter()

UPLOAD_DIR = "../data/uploads"
MAX_FILE_SIZE_MB = 50
ALLOWED_TYPES = {"application/pdf", "application/x-pdf"}
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload one or multiple PDF files.
    Extracts text, generates embeddings, and stores in FAISS.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []
    pdf_service = PDFService()
    vs_service = VectorStoreService()

    for file in files:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            results.append({"filename": file.filename, "status": "error", "detail": "Only PDF files are accepted."})
            continue

        # Read & validate size
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            results.append({
                "filename": file.filename,
                "status": "error",
                "detail": f"File exceeds {MAX_FILE_SIZE_MB}MB limit.",
            })
            continue

        # Save file
        file_id = str(uuid.uuid4())
        save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        with open(save_path, "wb") as f:
            f.write(content)

        try:
            # Extract text
            pages = pdf_service.extract_text(save_path)
            if not pages:
                results.append({"filename": file.filename, "status": "error", "detail": "Could not extract text. File may be scanned/encrypted."})
                continue

            # Chunk & embed
            chunks = pdf_service.chunk_text(pages, file.filename)
            vs_service.add_documents(chunks)

            results.append({
                "filename": file.filename,
                "file_id": file_id,
                "status": "success",
                "pages": len(pages),
                "chunks": len(chunks),
                "size_mb": round(size_mb, 2),
            })
            logger.info(f"Processed: {file.filename} | pages={len(pages)} chunks={len(chunks)}")

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({"filename": file.filename, "status": "error", "detail": str(e)})

    success_count = sum(1 for r in results if r["status"] == "success")
    return JSONResponse(content={
        "message": f"Processed {success_count}/{len(files)} file(s) successfully.",
        "results": results,
    })


@router.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        files = []
        if os.path.exists(UPLOAD_DIR):
            for fname in os.listdir(UPLOAD_DIR):
                if fname.endswith(".pdf"):
                    fpath = os.path.join(UPLOAD_DIR, fname)
                    size_mb = os.path.getsize(fpath) / (1024 * 1024)
                    # Strip UUID prefix
                    display_name = "_".join(fname.split("_")[1:]) if "_" in fname else fname
                    files.append({"filename": display_name, "size_mb": round(size_mb, 2)})
        return {"documents": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def clear_documents():
    """Remove all uploaded documents and clear the vector store"""
    try:
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

        vs_service = VectorStoreService()
        vs_service.clear()

        return {"message": "All documents and vector store cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
