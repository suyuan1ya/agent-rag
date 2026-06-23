"""反思评估工具 — 检索充分性判断、引文验证。"""

from __future__ import annotations

import json
from typing import Any

from .base import Tool, ToolResult, tool


def create_reflection_tools(
    llm_client: Any,
    llm_model: str,
) -> list[Tool]:
    """创建反思评估工具集。

    Args:
        llm_client: OpenAI 兼容客户端
        llm_model: 模型名称

    Returns:
        [evaluate_sufficiency, verify_citation]
    """

    @tool(
        name="evaluate_sufficiency",
        description="评估当前检索结果是否足够回答用户问题。"
                    "检查: 结果数量、最高相关度分数、内容覆盖度。"
                    "如果不够，应调用 rewrite_query 改写查询或 decompose_question 拆解问题。",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户原始问题",
                },
                "results_summary": {
                    "type": "string",
                    "description": "检索结果的摘要（结果数量、最高分、标题等）",
                },
            },
            "required": ["question", "results_summary"],
        },
    )
    def evaluate_sufficiency(question: str, results_summary: str) -> ToolResult:
        prompt = (
            "评估检索结果是否足以回答用户问题。\n\n"
            f"用户问题: {question}\n\n"
            f"检索结果摘要: {results_summary}\n\n"
            "请判断并输出 JSON:\n"
            '{"sufficient": true/false, '
            '"score": 0.0-1.0, '
            '"reasoning": "判断理由", '
            '"suggestion": "如果不够，建议下一步行动"}'
        )

        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(raw)
            return ToolResult(
                content=json.dumps(result, ensure_ascii=False, indent=2),
                metadata=result,
            )
        except Exception as exc:
            # fallback: 规则判断
            return ToolResult(
                content=json.dumps({
                    "sufficient": True,
                    "score": 0.5,
                    "reasoning": f"LLM 评估失败({exc})，默认认为结果可用",
                }, ensure_ascii=False),
                metadata={"sufficient": True, "score": 0.5},
                error=f"评估失败，使用默认值: {exc}",
            )

    @tool(
        name="verify_citation",
        description="验证一个事实性陈述是否有检索到的原文支持。"
                    "逐条比对，防止回答中包含未经证实的编造内容。",
        parameters={
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "需要验证的事实性陈述",
                },
                "source_texts": {
                    "type": "string",
                    "description": "相关的检索结果原文（用 --- 分隔多条）",
                },
            },
            "required": ["claim", "source_texts"],
        },
    )
    def verify_citation(claim: str, source_texts: str) -> ToolResult:
        prompt = (
            "验证以下陈述是否有提供的原文支持。\n\n"
            f"陈述: {claim}\n\n"
            f"原文:\n{source_texts[:3000]}\n\n"
            "输出 JSON:\n"
            '{"supported": true/false, '
            '"evidence": "支持的原文片段（如果有）", '
            '"confidence": 0.0-1.0, '
            '"explanation": "判断理由"}'
        )

        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(raw)
            return ToolResult(
                content=json.dumps(result, ensure_ascii=False, indent=2),
                metadata=result,
            )
        except Exception as exc:
            return ToolResult(
                content=json.dumps({
                    "supported": False,
                    "evidence": "",
                    "confidence": 0.0,
                    "explanation": f"验证失败: {exc}",
                }, ensure_ascii=False),
                error=str(exc),
            )

    return [evaluate_sufficiency, verify_citation]
