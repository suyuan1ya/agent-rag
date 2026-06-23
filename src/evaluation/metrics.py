"""检索质量评估指标：Recall@k, MRR, NDCG@k。"""

from __future__ import annotations

from collections import defaultdict


def generate_questions(
    rag, chunks: list[dict], num: int, seed: int = 42
) -> list[dict]:
    """用 LLM 为随机采样的 chunk 生成自然语言问题。"""
    import random

    random.seed(seed)
    sampled = random.sample(chunks, min(num, len(chunks)))
    test_cases = []

    prompt_template = (
        "阅读以下学术论文片段，生成一个可以用该片段回答的自然语言问题。"
        "问题应该是研究者可能提出的具体问题，而非泛泛而谈。"
        "只输出问题本身，不要加引号、编号或任何额外说明。\n\n"
        "论文片段：{text}"
    )

    for i, chunk in enumerate(sampled):
        try:
            response = rag.llm_client.chat.completions.create(
                model=rag.llm_model,
                messages=[{"role": "user", "content": prompt_template.format(text=chunk["text"])}],
                temperature=0.7,
            )
            question = response.choices[0].message.content.strip().strip('"').strip("'")
            test_cases.append({
                "question": question,
                "source_idx": i,
                "page": chunk["page_number"],
                "text": chunk["text"],
            })
            print(f"  [{len(test_cases)}/{num}] {question[:80]}...")
        except Exception:
            print(f"  [WARN] 生成问题失败 (chunk {i})")
            continue

    return test_cases


def compute_metrics(test_cases: list[dict], search_fn, top_k: int = 5) -> dict:
    """计算 Recall@k, MRR, NDCG@k。"""
    recalls = defaultdict(list)
    mrr_scores = []
    ndcg_scores = defaultdict(list)

    for case in test_cases:
        results = search_fn(case["question"], top_k=top_k) or []

        def is_relevant(result_text: str) -> bool:
            set_a, set_b = set(result_text), set(case["text"])
            if not set_a or not set_b:
                return False
            return len(set_a & set_b) / len(set_a | set_b) > 0.3

        hit_ranks = []
        for rank, r in enumerate(results):
            if is_relevant(r["text"]):
                hit_ranks.append(rank + 1)

        for k in [1, 3, 5]:
            if k <= top_k:
                recalls[k].append(1.0 if any(r <= k for r in hit_ranks) else 0.0)

        if hit_ranks:
            mrr_scores.append(1.0 / hit_ranks[0])
        else:
            mrr_scores.append(0.0)

        for k in [1, 3, 5]:
            if k <= top_k:
                dcg = sum(
                    1.0 / (2.0 ** (i + 2))
                    for i, r in enumerate(results[:k])
                    if is_relevant(r["text"])
                )
                idcg = 1.0 / (2.0 ** 2)
                ndcg_scores[k].append(dcg / idcg if idcg > 0 else 0.0)

    metrics = {}
    for k in [1, 3, 5]:
        if k <= top_k:
            metrics[f"Recall@{k}"] = (
                sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0.0
            )
            metrics[f"NDCG@{k}"] = (
                sum(ndcg_scores[k]) / len(ndcg_scores[k]) if ndcg_scores[k] else 0.0
            )
    metrics["MRR"] = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    metrics["queries"] = len(test_cases)

    return metrics


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  测试题数: {metrics['queries']}")
    for k in [1, 3, 5]:
        key_r = f"Recall@{k}"
        key_n = f"NDCG@{k}"
        if key_r in metrics:
            print(f"  {key_r:>10}: {metrics[key_r]:.3f}")
        if key_n in metrics:
            print(f"  {key_n:>10}: {metrics[key_n]:.3f}")
    print(f"  {'MRR':>10}: {metrics['MRR']:.3f}")
