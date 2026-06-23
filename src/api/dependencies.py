"""FastAPI 依赖注入 — Agent 单例管理、对话管理。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.agent.memory.conversation import ConversationBuffer

if TYPE_CHECKING:
    from src.agent.agent import ResearchAgent


# 全局 Agent 实例（模块级单例，应用启动时由 lifespan 初始化）
_agent: "ResearchAgent | None" = None
_agent_pdf: str = ""


class ConversationManager:
    """管理对话生命周期，支持 TTL 过期清理。"""

    def __init__(self):
        self._conversations: dict[str, ConversationBuffer] = {}
        self._last_cleanup: float = time.time()

    def get_or_create(self, conversation_id: str) -> ConversationBuffer:
        self._maybe_cleanup()

        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = ConversationBuffer(
                conversation_id=conversation_id,
                max_turns=20,
                ttl_seconds=3600,
            )
            return self._conversations[conversation_id]

        conv = self._conversations[conversation_id]
        if conv.is_expired():
            conv.clear()
        else:
            conv.touch()
        return conv

    def delete(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)

    def _maybe_cleanup(self) -> None:
        """每 5 分钟清理过期对话。"""
        now = time.time()
        if now - self._last_cleanup < 300:
            return
        expired = [
            cid for cid, conv in self._conversations.items()
            if conv.is_expired()
        ]
        for cid in expired:
            del self._conversations[cid]
        self._last_cleanup = now


# 全局对话管理器
conversation_manager = ConversationManager()


def get_agent() -> "ResearchAgent | None":
    """获取全局 Agent 实例。"""
    return _agent


def set_agent(agent: "ResearchAgent", pdf: str = "") -> None:
    """设置全局 Agent 实例（由 lifespan 调用）。"""
    global _agent, _agent_pdf
    _agent = agent
    _agent_pdf = pdf
