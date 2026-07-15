"""POST /api/v1/search — 直接调用知识库对应的 RAGEngine。"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from src.api.dependencies import get_agent, restore_agent
from src.api.schemas import SearchRequest, SearchResponse, Source

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    agent = get_agent(request.knowledge_base_id)
    if agent is None:
        agent = await asyncio.to_thread(restore_agent, request.knowledge_base_id)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未就绪，请先上传 PDF")

    rag = agent.rag
    strategy = request.strategy

    if strategy == "dense":
        results = await rag.dense_search(request.query, top_k=request.top_k)
    elif strategy == "keyword":
        results = rag.keyword_search(request.query, top_k=request.top_k)
    else:
        results = await rag.hybrid_search(request.query, top_k=request.top_k)

    if not results:
        return SearchResponse(results=[], query=request.query, strategy=strategy)

    sources = [Source(text=r["text"], page=r["page"], score=r["score"]) for r in results]

    return SearchResponse(
        results=sources,
        query=request.query,
        strategy=strategy,
    )
