"""Search tools — wrapping RAG engine methods as Agent-callable tools.

Compatible with both legacy RAGSystem (sync) and new RAGEngine (async).
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import Tool, ToolResult, tool


def create_search_tools(rag: Any) -> list[Tool]:
    """Create search tool set from a RAG instance.

    Args:
        rag: RAGSystem (legacy, sync) or RAGEngine (new, async).
             Detection is automatic based on method signatures.

    Returns:
        [dense_search, keyword_search, hybrid_search]
    """
    # Detect engine type by checking if hybrid_search is a coroutine function
    _is_async = asyncio.iscoroutinefunction(getattr(rag, "hybrid_search", None))

    @tool(
        name="dense_search",
        description="Semantic vector search (Dense Retrieval). Best for conceptual questions, "
        "definitions, and principle descriptions. Uses embeddings to find "
        "semantically similar content. Ideal when the query is about concepts "
        "rather than specific terms.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 5, "description": "Number of results"},
            },
            "required": ["query"],
        },
    )
    async def dense_search(query: str, top_k: int = 5) -> ToolResult:
        if _is_async:
            results = await rag.dense_search(query, top_k=top_k)
        else:
            results = await asyncio.to_thread(rag.search_similar, query, top_k=top_k)

        if not results:
            return ToolResult(
                content="No relevant results found. Try keyword_search for exact matches "
                "or hybrid_search for comprehensive retrieval.",
                metadata={"count": 0, "query": query},
            )
        return ToolResult(
            content=_format_results(results, "Semantic Search"),
            metadata={
                "count": len(results),
                "query": query,
                "strategy": "dense",
                "results": results,
            },
        )

    @tool(
        name="keyword_search",
        description="BM25 keyword search (Sparse Retrieval). Best for specific terms, "
        "abbreviations, proper nouns, and exact phrase matching. Uses TF-IDF "
        "to find documents containing specific keywords. Ideal for legal "
        "terminology, party names, dates, and article numbers.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords or phrases to search"},
                "top_k": {"type": "integer", "default": 5, "description": "Number of results"},
            },
            "required": ["query"],
        },
    )
    async def keyword_search(query: str, top_k: int = 5) -> ToolResult:
        if _is_async:
            results = rag.keyword_search(query, top_k=top_k)
        else:
            results = rag.keyword_search(query, top_k=top_k)

        if not results:
            return ToolResult(
                content="No keyword matches found. Try dense_search for semantic retrieval "
                "or hybrid_search for comprehensive retrieval.",
                metadata={"count": 0, "query": query},
            )
        return ToolResult(
            content=_format_results(results, "Keyword Search"),
            metadata={
                "count": len(results),
                "query": query,
                "strategy": "keyword",
                "results": results,
            },
        )

    @tool(
        name="hybrid_search",
        description="Hybrid search (Dense + Sparse). Combines semantic understanding with "
        "keyword matching via RRF (Reciprocal Rank Fusion) algorithm, then "
        "re-ranks with CrossEncoder. This is the DEFAULT and recommended "
        "strategy for most queries. Best overall recall and precision.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 5, "description": "Number of results"},
            },
            "required": ["query"],
        },
    )
    async def hybrid_search(query: str, top_k: int = 5) -> ToolResult:
        if _is_async:
            results = await rag.hybrid_search(query, top_k=top_k)
        else:
            results = await asyncio.to_thread(rag.hybrid_search, query, top_k=top_k)

        if not results:
            return ToolResult(
                content="Hybrid search returned no results. Try decomposing the question "
                "into smaller sub-questions using decompose_question.",
                metadata={"count": 0, "query": query},
            )
        return ToolResult(
            content=_format_results(results, "Hybrid Search"),
            metadata={
                "count": len(results),
                "query": query,
                "strategy": "hybrid",
                "results": results,
            },
        )

    return [dense_search, keyword_search, hybrid_search]


def _format_results(results: list[dict], label: str) -> str:
    """Format search results as LLM-readable text."""
    lines = [f"[{label}] {len(results)} results:\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] Page {r['page']} | Score {r['score']:.3f}\n    {r['text'][:500]}")
    return "\n".join(lines)
