"""GET /api/v1/health — AgentRAG 健康检查。"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import get_agent
from src.api.schemas import HealthResponse

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    agent = get_agent()
    if agent is None:
        return HealthResponse(
            status="no_document",
            milvus_connected=False,
            model_loaded=False,
        )

    rag = agent.rag
    return HealthResponse(
        status="ready",
        milvus_connected=rag.collection is not None,
        model_loaded=rag.embed_model is not None,
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
