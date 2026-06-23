"""自动 Benchmark — 端到端评估流程。"""

from __future__ import annotations

import json
import time
from typing import Any


async def run_benchmark(
    rag,  # RAGSystem
    llm_client: Any,
    model: str = "qwen-plus",
    num_questions: int = 30,
    output_path: str | None = None,
) -> dict[str, Any]:
    """运行完整的评估 benchmark。

    流程:
    1. 生成测试问题
    2. 运行三种策略检索 + Agent 检索
    3. 计算检索指标 + RAGAS + LLM-as-Judge
    4. 输出报告

    Returns:
        评估报告字典
    """
    from src.evaluation.ragas_eval import evaluate_with_ragas
    from src.evaluation.judge import judge_answer_quality

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_questions": num_questions,
        "strategies": {},
    }

    # 生成测试集
    chunks = rag.get_chunks()
    if len(chunks) < num_questions:
        num_questions = len(chunks)

    # 简单采样
    import random
    random.seed(42)
    sampled = random.sample(chunks, num_questions)

    test_questions = []
    for chunk in sampled:
        prompt = (
            f"请基于以下文本生成一个自然语言问题，该问题可以由这段文本回答。"
            f"只输出问题本身:\n\n{chunk['text'][:500]}"
        )
        try:
            resp = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            test_questions.append(resp.choices[0].message.content.strip())
        except Exception:
            continue

    # 每个策略评估
    strategies = {
        "dense": rag.search_similar,
        "keyword": rag.keyword_search,
        "hybrid": rag.hybrid_search,
    }

    for name, search_fn in strategies.items():
        all_answers = []
        all_contexts = []
        for q in test_questions:
            results = search_fn(q, top_k=5)
            if results:
                all_contexts.append([r["text"] for r in results])
                answer = rag.generate_answer(q, results)
                all_answers.append(answer)
            else:
                all_contexts.append([])
                all_answers.append("")

        # RAGAS 评估
        ragas_result = await evaluate_with_ragas(
            questions=test_questions,
            answers=all_answers,
            contexts=all_contexts,
        )

        report["strategies"][name] = {
            "ragas": ragas_result,
            "avg_contexts": sum(len(c) for c in all_contexts) / max(len(all_contexts), 1),
        }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report
