"""Agent tool abstraction — Tool protocol, ToolResult, and @tool decorator.

Supports both sync and async tools. Async tools are awaited directly;
sync tools are executed via asyncio.to_thread.
Uses OpenAI function calling schema for tool descriptions.
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ToolResult:
    """Result from a tool execution."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_message(self) -> dict:
        """Convert to OpenAI tool result message format."""
        return {"role": "tool", "content": self.content}


class Tool(Protocol):
    """Tool protocol — any object satisfying this can be registered.

    Supports both sync and async underlying functions.
    """

    name: str
    description: str
    parameters: dict  # JSON Schema
    _is_async: bool

    def __call__(self, **kwargs) -> ToolResult: ...
    async def acall(self, **kwargs) -> ToolResult: ...


def tool(
    name: str,
    description: str,
    parameters: dict | None = None,
) -> Callable:
    """Decorator: register a function as an Agent tool (sync or async).

    Usage:
        @tool(name="hybrid_search", description="Hybrid retrieval", parameters={...})
        async def hybrid_search(query: str, top_k: int = 5) -> ToolResult:
            ...
    """
    def decorator(fn: Callable):
        is_async = asyncio.iscoroutinefunction(fn)

        if is_async:
            @functools.wraps(fn)
            async def async_impl(**kwargs) -> ToolResult:
                try:
                    result = await fn(**kwargs)
                    if isinstance(result, ToolResult):
                        return result
                    return ToolResult(content=str(result))
                except Exception as exc:
                    return ToolResult(content="", error=f"{type(exc).__name__}: {exc}")

            wrapper = async_impl
            wrapper._is_async = True

            # Provide sync __call__ for backward compat
            @functools.wraps(fn)
            def sync_call(**kwargs) -> ToolResult:
                import threading
                result_container = []
                error_container = []

                def _run():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result_container.append(loop.run_until_complete(async_impl(**kwargs)))
                    except Exception as e:
                        error_container.append(e)
                    finally:
                        loop.close()

                thread = threading.Thread(target=_run)
                thread.start()
                thread.join()
                if error_container:
                    return ToolResult(content="", error=str(error_container[0]))
                return result_container[0]

            wrapper.__call__ = sync_call
            wrapper.acall = async_impl
        else:
            @functools.wraps(fn)
            def sync_impl(**kwargs) -> ToolResult:
                try:
                    result = fn(**kwargs)
                    if isinstance(result, ToolResult):
                        return result
                    return ToolResult(content=str(result))
                except Exception as exc:
                    return ToolResult(content="", error=f"{type(exc).__name__}: {exc}")

            wrapper = sync_impl
            wrapper._is_async = False
            wrapper.__call__ = sync_impl

            @functools.wraps(fn)
            async def async_wrapper(**kwargs) -> ToolResult:
                try:
                    result = await asyncio.to_thread(fn, **kwargs)
                    if isinstance(result, ToolResult):
                        return result
                    return ToolResult(content=str(result))
                except Exception as exc:
                    return ToolResult(content="", error=f"{type(exc).__name__}: {exc}")
            wrapper.acall = async_wrapper

        wrapper.name = name
        wrapper.description = description
        wrapper.parameters = parameters or {
            "type": "object", "properties": {}, "required": [],
        }
        return wrapper
    return decorator


def get_openai_tool_schema(t: Tool) -> dict:
    """Generate OpenAI function calling compatible schema from a Tool."""
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        },
    }
