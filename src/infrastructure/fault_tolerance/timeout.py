"""Timeout management — per-tool and global query timeouts."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class TimeoutError(Exception):
    """Raised when an operation exceeds its timeout."""
    def __init__(self, operation: str, timeout: float):
        super().__init__(f"Operation '{operation}' timed out after {timeout:.1f}s")
        self.operation = operation
        self.timeout = timeout


class TimeoutManager:
    """Manages timeouts for agent operations.

    Supports:
    - Per-tool timeouts (different tools have different expectations)
    - Global query timeout (overall query must complete within this)
    - Timeout warnings (log when approaching deadline)

    Usage:
        tm = TimeoutManager(global_timeout=120.0)
        tm.set_tool_timeout("hybrid_search", 10.0)
        tm.set_tool_timeout("llm_call", 30.0)

        result = await tm.execute("hybrid_search", search_fn, query, top_k=5)
    """

    def __init__(
        self,
        global_timeout: float = 120.0,
        default_tool_timeout: float = 15.0,
    ):
        self.global_timeout = global_timeout
        self.default_tool_timeout = default_tool_timeout
        self._tool_timeouts: dict[str, float] = {
            # Search tools — relatively fast
            "dense_search": 15.0,
            "keyword_search": 5.0,
            "hybrid_search": 20.0,
            # Query transformation tools — LLM calls
            "decompose_question": 15.0,
            "rewrite_query": 10.0,
            "generate_hypothetical_answer": 20.0,
            # Reflection tools — LLM calls
            "evaluate_sufficiency": 10.0,
            "verify_citation": 10.0,
            # LLM calls
            "llm_chat": 30.0,
            "llm_stream": 60.0,
        }
        self._query_start: float = 0.0

    def set_tool_timeout(self, tool_name: str, timeout: float) -> None:
        """Override the default timeout for a specific tool."""
        self._tool_timeouts[tool_name] = timeout

    def get_tool_timeout(self, tool_name: str) -> float:
        """Get the timeout for a specific tool."""
        return self._tool_timeouts.get(tool_name, self.default_tool_timeout)

    def start_query(self) -> None:
        """Mark the start of a new query (resets global timer)."""
        self._query_start = time.monotonic()

    def remaining_global_timeout(self) -> float:
        """Seconds remaining in the global query timeout."""
        if self._query_start == 0:
            return self.global_timeout
        elapsed = time.monotonic() - self._query_start
        return max(0.0, self.global_timeout - elapsed)

    def check_global_timeout(self) -> None:
        """Raise TimeoutError if global timeout is exceeded."""
        remaining = self.remaining_global_timeout()
        if remaining <= 0:
            raise TimeoutError("global_query", self.global_timeout)

    async def execute(
        self,
        operation: str,
        fn: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute fn with timeout protection.

        Args:
            operation: name of the operation (tool name or logical name)
            fn: async callable to execute
            *args, **kwargs: forwarded to fn

        Returns:
            fn's return value

        Raises:
            TimeoutError: if the operation exceeds its timeout
        """
        timeout = self.get_tool_timeout(operation)

        # Also cap by remaining global timeout
        global_remaining = self.remaining_global_timeout()
        effective_timeout = min(timeout, global_remaining)

        if effective_timeout <= 0:
            raise TimeoutError(operation, timeout)

        try:
            return await asyncio.wait_for(
                fn(*args, **kwargs),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(operation, timeout) from None

    def execute_sync(
        self,
        operation: str,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Synchronous variant with timeout."""
        import concurrent.futures

        timeout = self.get_tool_timeout(operation)
        global_remaining = self.remaining_global_timeout()
        effective_timeout = min(timeout, global_remaining)

        if effective_timeout <= 0:
            raise TimeoutError(operation, timeout)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=effective_timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(operation, timeout) from None
