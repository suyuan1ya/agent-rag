"""测试 Agent 工具：创建、Schema 生成、ToolRegistry 集成。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agent.tools.base import ToolResult, get_openai_tool_schema, tool
from src.agent.tools.query_tools import create_query_tools
from src.agent.tools.registry import ToolRegistry


class TestToolResult:
    def test_success_result(self):
        result = ToolResult(content="成功", metadata={"count": 5})
        assert result.success is True
        assert result.content == "成功"
        assert result.metadata["count"] == 5

    def test_error_result(self):
        result = ToolResult(content="", error="出错了")
        assert result.success is False
        assert result.error == "出错了"

    def test_defaults(self):
        result = ToolResult(content="")
        assert result.success is True
        assert result.metadata == {}


class TestToolDecorator:
    def test_decorator_creates_tool(self):
        @tool(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "A number"},
                },
                "required": ["x"],
            },
        )
        def my_tool(x: int) -> ToolResult:
            return ToolResult(content=str(x * 2))

        assert my_tool.name == "test_tool"
        assert my_tool.description == "A test tool"
        # Tool is callable directly (not via .call())
        result = my_tool(x=3)
        assert result.content == "6"

    def test_tool_exception_handling(self):
        @tool(
            name="broken",
            description="Always raises",
            parameters={"type": "object", "properties": {}},
        )
        def broken_tool() -> ToolResult:
            raise ValueError("boom")

        result = broken_tool()
        assert result.success is False
        assert "ValueError" in result.error
        assert "boom" in result.error

    def test_query_tool_awaits_async_llm_client(self):
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="改写后的查询"))]
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        rewrite_query = create_query_tools(client, "test-model")[1]

        result = asyncio.run(rewrite_query.acall(original_query="原始查询"))

        assert result.success is True
        assert result.content == "改写后的查询"
        create.assert_awaited_once()


class TestToolSchema:
    def test_get_openai_schema(self):
        @tool(
            name="search",
            description="Search documents",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        )
        def search(query: str, top_k: int = 5) -> ToolResult:
            return ToolResult(content="")

        schema = get_openai_tool_schema(search)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert "query" in schema["function"]["parameters"]["properties"]


class TestToolRegistry:
    @pytest.fixture
    def sample_tool(self):
        @tool(
            name="greet",
            description="Greet someone",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        def greet(name: str) -> ToolResult:
            return ToolResult(content=f"Hello {name}")

        return greet

    def test_register_tool(self, sample_tool):
        registry = ToolRegistry()
        registry.register(sample_tool)
        assert registry.get("greet") is sample_tool

    def test_register_duplicate_raises(self, sample_tool):
        registry = ToolRegistry()
        registry.register(sample_tool)
        with pytest.raises(ValueError):
            registry.register(sample_tool)

    def test_list_tools(self, sample_tool):
        registry = ToolRegistry()
        registry.register(sample_tool)
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0] == "greet"  # list_tools returns names

    def test_get_openai_schemas(self, sample_tool):
        registry = ToolRegistry()
        registry.register(sample_tool)
        schemas = registry.get_openai_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"

    def test_execute_success(self, sample_tool):
        registry = ToolRegistry()
        registry.register(sample_tool)
        # execute takes dict, not **kwargs
        result = registry.execute("greet", {"name": "World"})
        assert result.success is True
        assert "Hello World" in result.content

    def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent", {})
        assert result.success is False
