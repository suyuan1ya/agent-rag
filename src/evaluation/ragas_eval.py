"""RAGAS 评估集成 — faithfulness, answer_relevancy, context_precision, context_recall。"""

from __future__ import annotations

from typing import Any


async def evaluate_with_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> dict[str, Any]:
    """使用 RAGAS 评估 RAG 管道质量。

    Args:
        questions: 问题列表
        answers: 对应答案列表
        contexts: 每个答案的检索上下文列表
        ground_truths: 可选的真值答案列表

    Returns:
        {"faithfulness": float, "answer_relevancy": float, ...}
    """
    try:
        from ragas import evaluate as ragas_eval
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truths:
            metrics.append(context_recall)

        result = ragas_eval(dataset, metrics=metrics)
        return result  # type: ignore[return-value]
    except ImportError:
        return {"error": "ragas 未安装，请 pip install ragas"}
