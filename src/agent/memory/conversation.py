"""短期对话记忆 — 基于 deque 的最近 N 轮对话缓冲区，支持自动摘要和 TTL 过期。"""

from __future__ import annotations

import time
from collections import deque


class ConversationBuffer:
    """存储和管理多轮对话历史。

    特性:
    - 固定窗口：保留最近 max_turns 轮（每轮 = user + assistant）
    - TTL 过期：超过 ttl_seconds 未活动的对话自动标记为过期
    - 可生成对话历史摘要注入 system prompt
    """

    def __init__(
        self,
        conversation_id: str = "default",
        max_turns: int = 20,
        ttl_seconds: int = 3600,
    ):
        self.conversation_id = conversation_id
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds

        # 存储: [(role, content, timestamp), ...]
        self._messages: deque[tuple[str, str, float]] = deque()
        self._last_activity: float = time.time()

    def add(self, role: str, content: str) -> None:
        """添加一条消息。role ∈ {user, assistant, system, tool}。"""
        self._messages.append((role, content, time.time()))
        self._last_activity = time.time()

        # 超过最大轮数时移除最早的
        self._trim()

    def _trim(self) -> None:
        """保持消息数量在 max_turns * 2（每轮 user + assistant 计 2 条）以内。"""
        limit = self.max_turns * 2
        while len(self._messages) > limit:
            self._messages.popleft()

    def get_messages(self, max_turns: int | None = None) -> list[dict]:
        """获取对话历史，格式化为 OpenAI messages 数组。

        Args:
            max_turns: 限制返回最近 N 轮，None 表示使用配置的默认值

        Returns:
            [{"role": str, "content": str}, ...]
        """
        limit = (max_turns or self.max_turns) * 2
        recent = list(self._messages)[-limit:]
        return [
            {"role": role, "content": content}
            for role, content, _ in recent
        ]

    def get_last_n_turns(self, n: int) -> list[dict]:
        """获取最近 n 轮对话。"""
        return self.get_messages(max_turns=n)

    def is_expired(self) -> bool:
        """检查对话是否因不活跃而过期。"""
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self._last_activity) > self.ttl_seconds

    def touch(self) -> None:
        """刷新最后活动时间（防止过期）。"""
        self._last_activity = time.time()

    def clear(self) -> None:
        """清空对话历史。"""
        self._messages.clear()
        self._last_activity = time.time()

    def summarize(self, llm) -> str:
        """生成对话摘要（用于注入 system prompt）。

        Args:
            llm: LLMProvider 或兼容对象，需有 chat_sync 方法

        Returns:
            摘要字符串，对话为空时返回空字符串
        """
        if not self._messages:
            return ""

        history_text = "\n".join(
            f"{role}: {content[:200]}"
            for role, content, _ in self._messages
        )
        if len(history_text) > 3000:
            history_text = history_text[-3000:]

        prompt = (
            "请用 2-3 句话总结以下对话的核心内容和用户关注点。只输出总结:\n\n"
            f"{history_text}"
        )

        try:
            return llm.chat_sync(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
            )
        except Exception:
            return ""

    def __len__(self) -> int:
        return len(self._messages)
