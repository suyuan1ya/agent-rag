"""ToolRegistry — 工具注册、Schema 生成、执行调度。"""

from __future__ import annotations

from .base import Tool, ToolResult, get_openai_tool_schema


class ToolRegistry:
    """管理 Agent 可用工具的注册和调度。

    提供:
    - 工具注册（含去重）
    - 生成 OpenAI 兼容的 tools 数组
    - 按名称调度执行
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
        """生成 OpenAI function calling 兼容的工具列表。"""
        return [get_openai_tool_schema(t) for t in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> ToolResult:
        """按名称调度工具执行。

        Returns:
            ToolResult: 始终返回 ToolResult，即使工具不存在也返回带 error 的结果。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                content="",
                error=f"未知工具 '{name}'。可用工具: {', '.join(self._tools.keys())}",
            )
        return tool(**arguments)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
