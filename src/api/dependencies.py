"""In-process runtime registry with knowledge-base isolation."""

from __future__ import annotations

import threading
import time
from typing import Any

from src.agent.memory.conversation import ConversationBuffer
from src.core.config import get_settings


class RuntimeRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, knowledge_base_id: str = "default") -> Any | None:
        with self._lock:
            return self._agents.get(knowledge_base_id)

    def set(self, knowledge_base_id: str, agent: Any) -> None:
        with self._lock:
            previous = self._agents.get(knowledge_base_id)
            self._agents[knowledge_base_id] = agent
        if previous is not None and previous is not agent:
            previous.close()

    def values(self) -> list[Any]:
        with self._lock:
            return list(self._agents.values())

    def clear(self) -> None:
        with self._lock:
            agents, self._agents = list(self._agents.values()), {}
        for agent in agents:
            agent.close()


class ConversationManager:
    def __init__(self) -> None:
        self._conversations: dict[tuple[str, str], ConversationBuffer] = {}
        self._last_cleanup = time.time()
        self._lock = threading.RLock()

    def get_or_create(
        self, conversation_id: str, knowledge_base_id: str = "default"
    ) -> ConversationBuffer:
        self._maybe_cleanup()
        key = (knowledge_base_id, conversation_id)
        settings = get_settings()
        with self._lock:
            conv = self._conversations.get(key)
            if conv is None or conv.is_expired():
                conv = ConversationBuffer(
                    conversation_id=conversation_id,
                    max_turns=settings.conversation_max_turns,
                    ttl_seconds=settings.conversation_ttl,
                )
                self._conversations[key] = conv
            else:
                conv.touch()
            return conv

    def delete(self, conversation_id: str, knowledge_base_id: str = "default") -> None:
        with self._lock:
            self._conversations.pop((knowledge_base_id, conversation_id), None)

    def _maybe_cleanup(self) -> None:
        now = time.time()
        if now - self._last_cleanup < 300:
            return
        with self._lock:
            expired = [key for key, conv in self._conversations.items() if conv.is_expired()]
            for key in expired:
                del self._conversations[key]
            self._last_cleanup = now


runtime_registry = RuntimeRegistry()
conversation_manager = ConversationManager()
_restore_locks: dict[str, threading.Lock] = {}
_restore_locks_guard = threading.RLock()

# Backward-compatible aliases used by older callers/tests.
_agent: Any | None = None
_agent_pdf = ""


def get_agent(knowledge_base_id: str = "default") -> Any | None:
    return runtime_registry.get(knowledge_base_id)


def restore_agent(knowledge_base_id: str = "default") -> Any | None:
    """Rebuild an in-memory agent from an existing persistent knowledge base."""
    existing = runtime_registry.get(knowledge_base_id)
    if existing is not None:
        return existing

    from src.core.runtime import (
        create_agent,
        create_engine,
        normalize_knowledge_base_id,
    )

    knowledge_base_id = normalize_knowledge_base_id(knowledge_base_id)
    with _restore_locks_guard:
        restore_lock = _restore_locks.setdefault(knowledge_base_id, threading.Lock())

    with restore_lock:
        existing = runtime_registry.get(knowledge_base_id)
        if existing is not None:
            return existing

        settings = get_settings()
        engine = create_engine(knowledge_base_id, settings)
        try:
            if not engine.initialize(require_documents=True):
                engine.close()
                return None
            agent = create_agent(engine, settings)
            set_agent(agent, knowledge_base_id=knowledge_base_id)
            return agent
        except Exception:
            engine.close()
            raise


def set_agent(agent: Any, pdf: str = "", knowledge_base_id: str = "default") -> None:
    global _agent, _agent_pdf
    runtime_registry.set(knowledge_base_id, agent)
    if knowledge_base_id == "default":
        _agent, _agent_pdf = agent, pdf
