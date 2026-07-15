"""AgentRAG API 服务 — 自驱式 RAG 框架的 REST 接口。

启动:
    python cli.py serve
    uvicorn src.api.app:create_app --factory
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.middleware.correlation import CorrelationIDMiddleware
from src.api.middleware.rate_limiter import RateLimiter, RateLimitMiddleware
from src.api.routes.chat import router as chat_router
from src.api.routes.documents import router as documents_router
from src.api.routes.health import router as health_router
from src.api.routes.search import router as search_router
from src.core.config import __version__, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    logger.info("RAG Agent API starting...")
    yield
    from src.api.dependencies import runtime_registry

    runtime_registry.clear()
    logger.info("RAG Agent API stopped.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AgentRAG",
        description="自驱式 RAG 框架 — Agent 自主决策检索策略、查询改写、自纠错。"
        "支持 Dense/BM25/Hybrid 多策略检索 + Reranker 精排 + SSE 流式输出。",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        RateLimitMiddleware,
        limiter=RateLimiter(settings.rate_limit_per_minute),
    )
    app.add_middleware(CorrelationIDMiddleware)

    # 路由
    app.include_router(chat_router)
    app.include_router(search_router)
    app.include_router(documents_router)
    app.include_router(health_router)

    web_dir = Path(__file__).resolve().parent.parent / "web"
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.middleware("http")
    async def disable_web_asset_cache(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.get("/", include_in_schema=False)
    async def web_app():
        return FileResponse(web_dir / "index.html")

    return app
