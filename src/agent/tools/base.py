"""Agent 工具抽象 — Tool 协议、ToolResult 和 @tool 装饰器。

采用 OpenAI function calling schema 作为工具描述格式，确保与 Qwen/DeepSeek 等
OpenAI 兼容提供商无缝对接。
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ToolResult:
    """工具执行结果。"""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_message(self) -> dict:
        """转为 OpenAI tool result message 格式。"""
        return {
            "role": "tool",
            "content": self.content,
        }


class Tool(Protocol):
    """工具协议 — 任何满足此接口的对象都可注册到 ToolRegistry。"""

    name: str
    description: str
    parameters: dict  # JSON Schema

    def __call__(self, **kwargs) -> ToolResult:
        ...


def tool(
    name: str,
    description: str,
    parameters: dict | None = None,
) -> Callable:
    """装饰器：将普通函数注册为 Agent 工具。

    Usage:
        @tool(
            name="hybrid_search",
            description="混合检索文档",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        )
        async def hybrid_search(query: str, top_k: int = 5) -> ToolResult:
            ...
    """
    def decorator(fn: Callable) -> Tool:
        @functools.wraps(fn)
        def wrapper(**kwargs) -> ToolResult:
            try:
                result = fn(**kwargs)
                if isinstance(result, ToolResult):
                    return result
                return ToolResult(content=str(result))
            except Exception as exc:
                return ToolResult(
                    content="",
                    error=f"{type(exc).__name__}: {exc}",
                )

        wrapper.name = name
        wrapper.description = description
        wrapper.parameters = parameters or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        return wrapper
    return decorator


def get_openai_tool_schema(t: Tool) -> dict:
    """从 Tool 对象生成 OpenAI function calling 兼容的 tool schema。"""
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        },
    }
