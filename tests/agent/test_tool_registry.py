"""ToolRegistry 单元测试。"""

from __future__ import annotations

import pytest

from src.agent.tools.base import ToolResult, tool
from src.agent.tools.registry import ToolRegistry


class TestToolRegistry:
    def test_register_tool(self):
        registry = ToolRegistry()

        @tool(name="test_tool", description="A test tool")
        def test_tool(**kwargs) -> ToolResult:
            return ToolResult(content="success", metadata={"count": 1})

        registry.register(test_tool)
        assert "test_tool" in registry
        assert len(registry) == 1
        assert registry.list_tools() == ["test_tool"]

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()

        @tool(name="dup", description="First")
        def dup1(**kwargs) -> ToolResult:
            return ToolResult(content="1")

        registry.register(dup1)

        @tool(name="dup", description="Second")
        def dup2(**kwargs) -> ToolResult:
            return ToolResult(content="2")

        with pytest.raises(ValueError, match="已注册"):
            registry.register(dup2)

    def test_get_openai_schemas(self):
        registry = ToolRegistry()

        @tool(
            name="search",
            description="Search documents",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        def search(**kwargs) -> ToolResult:
            return ToolResult(content="ok")

        registry.register(search)
        schemas = registry.get_openai_schemas()

        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "search"

    def test_execute_success(self):
        registry = ToolRegistry()

        @tool(name="add", description="Add numbers")
        def add(a: int, b: int) -> ToolResult:
            return ToolResult(content=str(a + b), metadata={"result": a + b})

        registry.register(add)
        result = registry.execute("add", {"a": 3, "b": 4})

        assert result.success
        assert result.content == "7"
        assert result.metadata["result"] == 7

    def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent", {})

        assert not result.success
        assert "未知工具" in result.error

    def test_tool_error_handling(self):
        registry = ToolRegistry()

        @tool(name="failing", description="Always fails")
        def failing(**kwargs) -> ToolResult:
            raise ValueError("Something went wrong")

        registry.register(failing)
        result = registry.execute("failing", {})

        assert not result.success
        assert "ValueError" in result.error
