"""Safe PDF upload and asynchronous ingestion endpoints."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from src.api.dependencies import set_agent
from src.core.config import get_settings
from src.core.runtime import create_agent, create_engine, normalize_knowledge_base_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["documents"])
_processing_status: dict[str, dict] = {}
_status_lock = threading.RLock()
_ingestion_locks: dict[str, threading.Lock] = {}


def _set_status(document_id: str, status: dict) -> None:
    with _status_lock:
        _processing_status[document_id] = status


def _public_processing_error(exc: Exception) -> str:
    message = str(exc)
    if "numpy.dtype size changed" in message:
        return "模型依赖版本不兼容，请重新安装项目依赖"
    if "Failed to load embedding model" in message:
        return "Embedding 模型加载失败，请检查网络或模型缓存"
    if "No valid text" in message:
        return "PDF 中没有可提取的文字，请换用文字版 PDF"
    if isinstance(exc, FileNotFoundError):
        return "上传的临时文件不存在，请重新上传"
    return f"{type(exc).__name__}: {message}"[:300]


@router.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    knowledge_base_id: str = Form("default"),
):
    settings = get_settings()
    try:
        knowledge_base_id = normalize_knowledge_base_id(knowledge_base_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    document_id = uuid.uuid4().hex[:12]
    original_filename = file.filename
    fd, tmp_path = tempfile.mkstemp(prefix=f"rag_{document_id}_", suffix=".pdf")
    size = 0
    header = b""
    try:
        with os.fdopen(fd, "wb") as target:
            while chunk := await file.read(1024 * 1024):
                if not header:
                    header = chunk[:5]
                size += len(chunk)
                if size > settings.upload_max_bytes:
                    raise HTTPException(status_code=413, detail="PDF 文件过大")
                target.write(chunk)
        if header != b"%PDF-":
            raise HTTPException(status_code=400, detail="文件内容不是有效的 PDF")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    finally:
        await file.close()

    _set_status(
        document_id,
        {"status": "processing", "chunk_count": 0, "knowledge_base_id": knowledge_base_id},
    )

    def process_document() -> None:
        engine = None
        try:
            with _status_lock:
                ingestion_lock = _ingestion_locks.setdefault(knowledge_base_id, threading.Lock())
            with ingestion_lock:
                _set_status(
                    document_id,
                    {
                        "status": "processing",
                        "stage": "loading_model",
                        "chunk_count": 0,
                        "knowledge_base_id": knowledge_base_id,
                    },
                )
                engine = create_engine(knowledge_base_id, settings)
                engine.initialize()
                _set_status(
                    document_id,
                    {
                        "status": "processing",
                        "stage": "indexing",
                        "chunk_count": 0,
                        "knowledge_base_id": knowledge_base_id,
                    },
                )
                chunk_count = asyncio.run(engine.ingest(tmp_path, source_name=original_filename))
                agent = create_agent(engine, settings)
                set_agent(
                    agent,
                    pdf=original_filename or "",
                    knowledge_base_id=knowledge_base_id,
                )
                _set_status(
                    document_id,
                    {
                        "status": "completed",
                        "chunk_count": chunk_count,
                        "knowledge_base_id": knowledge_base_id,
                    },
                )
                engine = None  # ownership transferred to the agent
        except Exception as exc:
            logger.exception("Document processing failed", extra={"document_id": document_id})
            error = _public_processing_error(exc)
            _set_status(
                document_id,
                {
                    "status": "failed",
                    "error": error,
                    "knowledge_base_id": knowledge_base_id,
                },
            )
        finally:
            if engine is not None:
                engine.close()
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass

    background_tasks.add_task(process_document)
    return {
        "document_id": document_id,
        "filename": file.filename,
        "knowledge_base_id": knowledge_base_id,
        "status": "processing",
    }


@router.get("/documents/{document_id}/status")
async def get_document_status(document_id: str):
    with _status_lock:
        status = _processing_status.get(document_id)
    if status is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"document_id": document_id, **status}
