"""ToolRegistry — tool registration, schema generation, async dispatch."""

from __future__ import annotations

from .base import Tool, ToolResult, get_openai_tool_schema


class ToolRegistry:
    """Manages registration and dispatch of Agent tools.

    Provides:
    - Tool registration (with dedup)
    - OpenAI-compatible tool schema generation
    - Sync and async dispatch by name
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"工具 '{t.name}' 已注册")
        self._tools[t.name] = t

    def register_many(self, tools: list[Tool]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_openai_schemas(self) -> list[dict]:
        """Generate OpenAI function calling compatible tool list."""
        return [get_openai_tool_schema(t) for t in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> ToolResult:
        """Synchronous tool dispatch.

        Returns ToolResult even if the tool is not found (with error).
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                content="",
                error=f"未知工具 '{name}'。可用工具: {', '.join(self._tools.keys())}",
            )
        return tool(**arguments)

    async def execute_async(self, name: str, arguments: dict) -> ToolResult:
        """Async tool dispatch — uses acall() for all tools.

        For sync-native tools, acall wraps them in asyncio.to_thread.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                content="",
                error=f"未知工具 '{name}'。可用工具: {', '.join(self._tools.keys())}",
            )
        return await tool.acall(**arguments)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
