"""Application composition root for the RAG engine and agent."""

from __future__ import annotations

import os
import re

from src.agent.orchestrator import AgentOrchestrator
from src.core.config import Settings, get_settings
from src.infrastructure.rag import RAGEngine


def normalize_knowledge_base_id(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]", "-", value.strip())
    if not value or len(value) > 64:
        raise ValueError("knowledge_base_id must contain 1-64 safe characters")
    return value


def create_engine(
    knowledge_base_id: str = "default",
    settings: Settings | None = None,
) -> RAGEngine:
    settings = settings or get_settings()
    kb_id = normalize_knowledge_base_id(knowledge_base_id)
    os.makedirs(settings.data_path, exist_ok=True)
    engine = RAGEngine(
        chroma_path=settings.chroma_path,
        collection_name=f"kb_{kb_id}",
        embedding_dim=settings.embedding_dim,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
        rrf_dense_weight=settings.rrf_dense_weight,
        rrf_sparse_weight=settings.rrf_sparse_weight,
    )
    engine.bm25.cache_path = os.path.join(settings.data_path, f"{kb_id}.bm25.json")
    return engine


def create_agent(
    engine: RAGEngine,
    settings: Settings | None = None,
) -> AgentOrchestrator:
    settings = settings or get_settings()
    agent = AgentOrchestrator(
        engine,
        max_iterations=settings.max_agent_iterations,
        settings=settings,
    )
    agent.initialize()
    return agent
