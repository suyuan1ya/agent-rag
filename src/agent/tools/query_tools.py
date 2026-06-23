"""查询变换工具 — 问题分解、查询改写、HyDE 假设答案生成。"""

from __future__ import annotations

import json
from typing import Any

from .base import Tool, ToolResult, tool


def create_query_tools(
    llm_client: Any,
    llm_model: str,
) -> list[Tool]:
    """创建查询变换工具集。

    Args:
        llm_client: OpenAI 兼容客户端
        llm_model: 模型名称

    Returns:
        [decompose_question, rewrite_query, generate_hypothetical_answer]
    """

    @tool(
        name="decompose_question",
        description="将复杂问题拆解为 1-4 个独立的子问题，便于分别检索。"
                    "适用于多跳推理、比较分析等需要多步检索的场景。",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "需要拆解的原始问题",
                },
            },
            "required": ["question"],
        },
    )
    def decompose_question(question: str) -> ToolResult:
        prompt = (
            "将以下复杂问题拆解为 1-4 个可独立检索的子问题。"
            "每个子问题必须独立、可直接用于检索，不要有依赖关系。"
            "如果原始问题已经很简单，只返回 1 个子问题。\n\n"
            f"问题: {question}\n\n"
            '只输出 JSON 数组: [{"question": "...", "rationale": "..."}]'
        )

        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            sub_queries = json.loads(raw)
            return ToolResult(
                content=json.dumps(sub_queries, ensure_ascii=False, indent=2),
                metadata={"count": len(sub_queries), "sub_queries": sub_queries},
            )
        except Exception as exc:
            return ToolResult(
                content=json.dumps(
                    [{"question": question, "rationale": "原始问题"}],
                    ensure_ascii=False,
                ),
                metadata={"count": 1},
                error=f"拆解失败，使用原始问题: {exc}",
            )

    @tool(
        name="rewrite_query",
        description="改写检索查询以获得更好的检索结果。可以展开缩写、添加同义词、"
                    "从不同角度重新表述。当检索结果不理想时使用。",
        parameters={
            "type": "object",
            "properties": {
                "original_query": {
                    "type": "string",
                    "description": "原始查询语句",
                },
                "context": {
                    "type": "string",
                    "description": "可选：之前检索结果的简要总结，帮助理解为什么需要改写",
                },
            },
            "required": ["original_query"],
        },
    )
    def rewrite_query(original_query: str, context: str = "") -> ToolResult:
        hint = ""
        if context:
            hint = f"\n之前的检索结果摘要: {context[:300]}"

        prompt = (
            f"原始查询: 「{original_query}」{hint}\n\n"
            "请改写此查询以获得更好的检索结果。你可以:\n"
            "- 展开缩写和专业术语\n"
            "- 添加同义词或相关表达\n"
            "- 从不同角度重新表述\n"
            "- 拆分为更具体的关键词组合\n\n"
            "只输出改写后的查询语句，不要加任何额外说明。"
        )

        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            rewritten = response.choices[0].message.content.strip().strip('"').strip("'")
            return ToolResult(
                content=rewritten,
                metadata={"original": original_query, "rewritten": rewritten},
            )
        except Exception as exc:
            return ToolResult(
                content=original_query,
                metadata={"original": original_query},
                error=f"改写失败: {exc}",
            )

    @tool(
        name="generate_hypothetical_answer",
        description="生成假设性答案（HyDE 策略）。先让 LLM 草拟一个可能的答案，"
                    "然后用这个假设答案做检索查询。假设答案与真实文档的语义更接近，"
                    "比直接用问题检索效果更好。适用于概念解释类问题。",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "需要回答的问题",
                },
            },
            "required": ["question"],
        },
    )
    def generate_hypothetical_answer(question: str) -> ToolResult:
        prompt = (
            f"请针对以下问题，草拟一个假设性的学术回答（200-300字）。"
            f"不需要完全准确，只需写出你期望在论文中找到的那种回答风格和内容。\n\n"
            f"问题: {question}\n\n"
            f"假设性回答:"
        )

        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            hypothetical = response.choices[0].message.content.strip()
            return ToolResult(
                content=hypothetical,
                metadata={"question": question, "length": len(hypothetical)},
            )
        except Exception as exc:
            return ToolResult(
                content=question,
                error=f"HyDE 生成失败: {exc}",
            )

    return [decompose_question, rewrite_query, generate_hypothetical_answer]
