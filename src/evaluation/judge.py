"""LLM-as-Judge — 用 LLM 评估答案质量。"""

from __future__ import annotations

import json
from typing import Any


async def judge_answer_quality(
    question: str,
    answer: str,
    reference_contexts: list[str],
    llm_client: Any,
    model: str = "qwen-plus",
    criteria: list[str] | None = None,
) -> dict[str, float]:
    """使用 LLM 评估答案质量。

    Args:
        question: 原始问题
        answer: Agent 生成的答案
        reference_contexts: 检索到的参考上下文
        llm_client: OpenAI 兼容客户端
        model: 模型名称
        criteria: 评估维度列表

    Returns:
        {"accuracy": 0.85, "completeness": 0.72, ...}
    """
    if criteria is None:
        criteria = ["accuracy", "completeness", "citation_quality", "conciseness"]

    context_text = "\n---\n".join(ctx[:500] for ctx in reference_contexts[:5])

    criteria_desc = "\n".join(
        f"- {c}: 1-10 分" for c in criteria
    )

    prompt = (
        "你是一个严格的评估者。请基于提供的参考文档，评估以下答案的质量。\n\n"
        f"问题: {question}\n\n"
        f"答案: {answer}\n\n"
        f"参考文档:\n{context_text[:3000]}\n\n"
        f"评估维度:\n{criteria_desc}\n\n"
        "输出 JSON: {\"accuracy\": 分数, \"completeness\": 分数, ...}"
        "每个维度给出 0.0-1.0 的分数。"
    )

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(raw)
    except Exception:
        return {c: 0.0 for c in criteria}
