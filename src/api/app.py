"""AgentRAG API 服务 — 自驱式 RAG 框架的 REST 接口。

启动:
    python cli.py serve
    uvicorn src.api.app:create_app --factory
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.chat import router as chat_router
from src.api.routes.search import router as search_router
from src.api.routes.documents import router as documents_router
from src.api.routes.health import router as health_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    logger.info("RAG Agent API starting...")
    yield
    # 关闭时清理
    from src.api.dependencies import get_agent
    agent = get_agent()
    if agent is not None:
        agent.close()
    logger.info("RAG Agent API stopped.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AgentRAG",
        description="自驱式 RAG 框架 — Agent 自主决策检索策略、查询改写、自纠错。"
                    "支持 Dense/BM25/Hybrid 多策略检索 + Reranker 精排 + SSE 流式输出。",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 路由
    app.include_router(chat_router)
    app.include_router(search_router)
    app.include_router(documents_router)
    app.include_router(health_router)

    return app
