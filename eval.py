"""RAG 检索质量评估：LLM 自动生成测试题 + Recall/MRR/NDCG 指标。

【修复7】改为导入 RAGSystem 类实例，不再依赖全局变量。

用法:
    python eval.py                  # 默认 30 条测试题
    python eval.py --num 50         # 50 条
    python eval.py --compare        # 对比 dense vs hybrid
"""

import argparse
import json
import random
import sys
import traceback
from collections import defaultdict

from rag import RAGSystem


def generate_questions(rag: RAGSystem, chunks: list[dict], num: int, seed: int = 42) -> list[dict]:
    """用 LLM 为随机采样的 chunk 生成自然语言问题。"""
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
            print(f"  ⚠️ 生成问题失败:")
            traceback.print_exc()
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
                dcg = sum(1.0 / (2.0 ** (i + 2)) for i, r in enumerate(results[:k])
                         if is_relevant(r["text"]))
                idcg = 1.0 / (2.0 ** 2)
                ndcg_scores[k].append(dcg / idcg if idcg > 0 else 0.0)

    metrics = {}
    for k in [1, 3, 5]:
        if k <= top_k:
            metrics[f"Recall@{k}"] = sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0
            metrics[f"NDCG@{k}"] = sum(ndcg_scores[k]) / len(ndcg_scores[k]) if ndcg_scores[k] else 0
    metrics["MRR"] = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
    metrics["queries"] = len(test_cases)

    return metrics


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  测试题数: {metrics['queries']}")
    for k in [1, 3, 5]:
        key_r = f"Recall@{k}"
        key_n = f"NDCG@{k}"
        if key_r in metrics:
            print(f"  {key_r:>10}: {metrics[key_r]:.3f}")
        if key_n in metrics:
            print(f"  {key_n:>10}: {metrics[key_n]:.3f}")
    print(f"  {'MRR':>10}: {metrics['MRR']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="RAG 检索质量评估")
    parser.add_argument("--num", type=int, default=30, help="生成测试题数量 (默认 30)")
    parser.add_argument("--compare", action="store_true", help="对比 dense / keyword / hybrid 三种策略")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default=None, help="保存测试题和结果到 JSON")
    parser.add_argument("--pdf", type=str, required=True, help="PDF 文件路径")
    args = parser.parse_args()

    print("初始化 RAG 环境...")
    rag = RAGSystem(pdf_path=args.pdf)

    try:
        rag.setup_milvus()
        rag.setup_models()
        rag.ingest_pdf()
    except Exception:
        print("❌ 初始化失败:")
        traceback.print_exc()
        rag.close()
        sys.exit(1)

    chunks = rag.get_chunks()
    if len(chunks) == 0:
        print("❌ 没有可评估的 chunk，请先入库 PDF")
        rag.close()
        sys.exit(1)

    if args.num > len(chunks):
        args.num = len(chunks)
        print(f"⚠️ chunk 总数 ({len(chunks)}) 不足，改为 {args.num} 条")

    print(f"\n生成 {args.num} 条测试题...")
    test_cases = generate_questions(rag, chunks, args.num, seed=args.seed)

    if args.compare:
        print_metrics("Dense (纯向量)", compute_metrics(test_cases, rag.search_similar))
        print_metrics("Sparse (BM25)", compute_metrics(test_cases, rag.keyword_search))
        print_metrics("Hybrid (Dense + Sparse)", compute_metrics(test_cases, rag.hybrid_search))
    else:
        print_metrics("Hybrid 检索评估", compute_metrics(test_cases, rag.hybrid_search))

    if args.output:
        output_data = {
            "config": {"num_questions": args.num, "seed": args.seed, "total_chunks": len(chunks)},
            "test_cases": [{"question": tc["question"], "page": tc["page"]} for tc in test_cases],
        }
        if args.compare:
            output_data["dense"] = compute_metrics(test_cases, rag.search_similar)
            output_data["sparse"] = compute_metrics(test_cases, rag.keyword_search)
            output_data["hybrid"] = compute_metrics(test_cases, rag.hybrid_search)
        else:
            output_data["hybrid"] = compute_metrics(test_cases, rag.hybrid_search)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n📁 评估结果已保存到 {args.output}")

    rag.close()
    print("\n评估完成。比较不同策略用: python eval.py --compare")


if __name__ == "__main__":
    main()
