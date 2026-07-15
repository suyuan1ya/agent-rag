"""GET /api/v1/health — AgentRAG 健康检查。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from src.api.dependencies import get_agent, restore_agent
from src.api.schemas import HealthResponse
from src.core.config import __version__

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(knowledge_base_id: str = "default"):
    agent = get_agent(knowledge_base_id)
    if agent is None:
        agent = await asyncio.to_thread(restore_agent, knowledge_base_id)
    if agent is None:
        return HealthResponse(
            status="no_document",
            vector_store_connected=False,
            model_loaded=False,
            version=__version__,
        )

    rag = agent.rag
    status = rag.degradation_status
    return HealthResponse(
        status="ready",
        version=__version__,
        vector_store_connected=status["vector_docs"] >= 0,
        model_loaded=getattr(rag.embedding, "_model", None) is not None,
        indexed_chunks=status["vector_docs"],
    )


@router.get("/metrics")
async def metrics():
    """Prometheus 指标端点（占位）。"""
    return {
        "agent": {
            "conversations_active": 0,
            "tools_registered": 0,
        },
    }
