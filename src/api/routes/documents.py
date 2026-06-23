"""POST /api/v1/documents/upload — PDF 上传和异步入库。"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile

from src.api.dependencies import get_agent, set_agent
from src.agent.agent import ResearchAgent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["documents"])

# 存储正在处理的任务状态
_processing_status: dict[str, dict] = {}


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    document_id = str(uuid.uuid4())[:8]

    # 保存上传文件到临时位置
    tmp_path = os.path.join(tempfile.gettempdir(), f"rag_agent_{document_id}.pdf")
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {exc}")

    _processing_status[document_id] = {"status": "processing", "chunk_count": 0}

    # 构造函数用于后台任务
    def process_document():
        try:
            from rag import RAGSystem
            rag = RAGSystem(pdf_path=tmp_path)
            rag.setup_milvus()
            rag.setup_models()
            rag.ingest_pdf()

            agent = ResearchAgent(rag)
            agent.initialize()
            set_agent(agent, pdf=tmp_path)

            chunks = rag.get_chunks()
            _processing_status[document_id] = {
                "status": "completed",
                "chunk_count": len(chunks),
            }
        except Exception as exc:
            logger.error(f"Document processing failed: {exc}")
            _processing_status[document_id] = {
                "status": "failed",
                "error": str(exc),
            }

    background_tasks.add_task(process_document)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "status": "processing",
    }


@router.get("/documents/{document_id}/status")
async def get_document_status(document_id: str):
    status = _processing_status.get(document_id)
    if status is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"document_id": document_id, **status}
