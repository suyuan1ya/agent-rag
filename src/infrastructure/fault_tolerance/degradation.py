"""Graceful Degradation Manager — registered fallback paths for component failures.

When a component fails (e.g., Reranker OOM, ChromaDB connection lost), the
DegradationManager routes to a pre-registered fallback instead of crashing.

Degradation paths are registered by context name and exception type:
  - "reranker": RerankerOOMError → raw RRF scores
  - "chromadb": ConnectionError → BM25-only mode
  - "embedding": RuntimeError → BM25-only mode
  - "llm_api": APIError → cached response or error message

All degradation events are logged with structured metadata for observability.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class DegradationEvent:
    """Recorded when a degradation path is activated."""

    def __init__(self, context: str, error_type: str, error_message: str):
        self.context = context
        self.error_type = error_type
        self.error_message = error_message


class DegradationManager:
    """Registry of degradation paths with automatic fallback routing.

    Usage:
        dm = DegradationManager()

        # Register fallback paths
        dm.register(
            "reranker", RerankerOOMError,
            lambda query, candidates, top_k: raw_rrf_results(candidates, top_k)
        )

        # Execute with automatic degradation
        result = await dm.execute(
            primary=lambda: reranker.rerank(query, candidates, top_k),
            context="reranker",
        )
    """

    def __init__(self):
        self._paths: dict[str, list[_DegradationPath]] = {}
        self._events: list[DegradationEvent] = []
        self._degraded_contexts: set[str] = set()

    def register(
        self,
        context: str,
        exception_type: type[Exception],
        fallback: Callable[..., Any],
        description: str = "",
    ) -> None:
        """Register a degradation path.

        Args:
            context: logical component name ("reranker", "chromadb", "llm_api", etc.)
            exception_type: exception class that triggers this fallback
            fallback: callable invoked when the exception is caught (receives same args as primary)
            description: human-readable description of the fallback behavior
        """
        if context not in self._paths:
            self._paths[context] = []
        self._paths[context].append(
            _DegradationPath(exception_type, fallback, description)
        )

    async def execute(
        self,
        primary: Callable[..., Awaitable[T]],
        context: str,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute primary() with automatic degradation on failure.

        If primary raises an exception matching a registered degradation path
        for the given context, the fallback is invoked instead.

        Args:
            primary: async callable (the primary implementation)
            context: degradation context name
            *args, **kwargs: forwarded to both primary and fallback

        Returns:
            primary's return value on success, fallback's return value on matched failure

        Raises:
            Original exception if no degradation path matches
        """
        try:
            return await primary(*args, **kwargs)
        except Exception as exc:
            paths = self._paths.get(context, [])
            for path in paths:
                if isinstance(exc, path.exception_type):
                    event = DegradationEvent(
                        context=context,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:200],
                    )
                    self._events.append(event)
                    self._degraded_contexts.add(context)

                    logger.warning(
                        "degradation_activated",
                        extra={
                            "context": context,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:200],
                            "fallback": path.description,
                        },
                    )

                    # Call the fallback
                    result = path.fallback(*args, **kwargs)
                    if hasattr(result, "__await__"):
                        return await result
                    return result

            # No matching degradation path → re-raise
            raise

    def execute_sync(
        self,
        primary: Callable[..., T],
        context: str,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Synchronous variant."""
        try:
            return primary(*args, **kwargs)
        except Exception as exc:
            paths = self._paths.get(context, [])
            for path in paths:
                if isinstance(exc, path.exception_type):
                    event = DegradationEvent(
                        context=context,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:200],
                    )
                    self._events.append(event)
                    self._degraded_contexts.add(context)
                    logger.warning(
                        "degradation_activated",
                        extra={
                            "context": context,
                            "error_type": type(exc).__name__,
                        },
                    )
                    return path.fallback(*args, **kwargs)
            raise

    # ── introspection ───────────────────────────────────────

    def is_degraded(self, context: str | None = None) -> bool:
        """Check if a context (or any) is currently degraded."""
        if context is not None:
            return context in self._degraded_contexts
        return len(self._degraded_contexts) > 0

    def clear_degradation(self, context: str | None = None) -> None:
        """Reset degradation state (e.g., after dependency recovers)."""
        if context is not None:
            self._degraded_contexts.discard(context)
        else:
            self._degraded_contexts.clear()

    @property
    def degraded_contexts(self) -> set[str]:
        return set(self._degraded_contexts)

    @property
    def events(self) -> list[DegradationEvent]:
        return list(self._events)

    @property
    def status(self) -> dict:
        return {
            "degraded": len(self._degraded_contexts) > 0,
            "degraded_contexts": sorted(self._degraded_contexts),
            "total_degradation_events": len(self._events),
            "recent_events": [
                {"context": e.context, "error": e.error_type}
                for e in self._events[-5:]
            ],
        }


class _DegradationPath:
    """Internal: a single degradation path entry."""

    def __init__(
        self,
        exception_type: type[Exception],
        fallback: Callable,
        description: str,
    ):
        self.exception_type = exception_type
        self.fallback = fallback
        self.description = description or f"Fallback on {exception_type.__name__}"
