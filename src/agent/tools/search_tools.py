"""检索工具 — 将 RAGSystem 的检索方法封装为 Agent 可调用的工具。"""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolResult, tool


def create_search_tools(rag: Any) -> list[Tool]:
    """基于 RAGSystem 实例创建检索工具集。

    Args:
        rag: RAGSystem 实例（需已初始化 Milvus + 模型 + PDF）

    Returns:
        [dense_search, keyword_search, hybrid_search]
    """

    @tool(
        name="dense_search",
        description="语义向量检索 (Dense Retrieval)。适用于概念解释、定义、原理描述等语义类问题。"
                    "使用 Embedding 模型理解查询意图，找到语义相近的内容段落。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询语句",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "返回结果数量，默认 5",
                },
            },
            "required": ["query"],
        },
    )
    def dense_search(query: str, top_k: int = 5) -> ToolResult:
        results = rag.search_similar(query, top_k=top_k)
        if not results:
            return ToolResult(
                content="未找到相关结果。建议尝试使用 keyword_search 进行精确匹配，"
                        "或使用 hybrid_search 进行综合检索。",
                metadata={"count": 0, "query": query},
            )
        return ToolResult(
            content=_format_results(results, "语义检索"),
            metadata={"count": len(results), "query": query, "strategy": "dense"},
        )

    @tool(
        name="keyword_search",
        description="BM25 关键词检索 (Sparse Retrieval)。适用于特定术语、缩写、专有名词、人名等精确匹配场景。"
                    "基于词频-逆文档频率 (TF-IDF) 进行检索，查找明确提及的关键词。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词或短语",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "返回结果数量，默认 5",
                },
            },
            "required": ["query"],
        },
    )
    def keyword_search(query: str, top_k: int = 5) -> ToolResult:
        results = rag.keyword_search(query, top_k=top_k)
        if not results:
            return ToolResult(
                content="未找到匹配关键词的结果。建议尝试使用 dense_search 进行语义检索，"
                        "或使用 hybrid_search 进行综合检索。",
                metadata={"count": 0, "query": query},
            )
        return ToolResult(
            content=_format_results(results, "关键词检索"),
            metadata={"count": len(results), "query": query, "strategy": "keyword"},
        )

    @tool(
        name="hybrid_search",
        description="混合检索 (Hybrid: Dense + Sparse)。综合语义理解和关键词匹配的优势，"
                    "通过 RRF (Reciprocal Rank Fusion) 算法融合两种结果，再用 CrossEncoder Reranker 精排。"
                    "适用于大多数查询场景，是 Agent 的默认检索策略。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询语句",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "返回结果数量，默认 5",
                },
            },
            "required": ["query"],
        },
    )
    def hybrid_search(query: str, top_k: int = 5) -> ToolResult:
        results = rag.hybrid_search(query, top_k=top_k)
        if not results:
            return ToolResult(
                content="混合检索无结果。建议尝试将问题拆解为更小的子问题逐一检索。",
                metadata={"count": 0, "query": query},
            )
        return ToolResult(
            content=_format_results(results, "混合检索"),
            metadata={"count": len(results), "query": query, "strategy": "hybrid"},
        )

    return [dense_search, keyword_search, hybrid_search]


def _format_results(results: list[dict], label: str) -> str:
    """格式化检索结果为 LLM 可读的文本。"""
    lines = [f"【{label}】共 {len(results)} 条结果:\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] 页码{r['page']} | 相关度 {r['score']:.3f}\n"
            f"    {r['text'][:500]}"
        )
    return "\n".join(lines)
