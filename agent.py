"""Research Agent —— 具备规划、工具选择、自纠错能力的论文问答智能体。

在 RAGSystem 之上增加 Agent 循环：
  1. 问题分解（Planning）
  2. 检索策略选择（Tool-use）
  3. 结果评估 + Query 改写重试（Self-correction）
  4. 多轮结果综合（Synthesis）

用法:
    python agent.py                          # 交互式问答
    python agent.py --query "这篇文章的主要贡献"  # 单次提问
"""

import argparse
import json
import re
import sys
import time
import traceback

from rag import RAGSystem


def _sanitize(text: str) -> str:
    """移除可能被 LLM 误解析为指令的特殊模式。"""
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'```\w*', '', text)
    text = re.sub(r'\[system\]|\[/system\]|\[user\]|\[/user\]|\[assistant\]|\[/assistant\]', '', text, flags=re.IGNORECASE)
    return text.strip()


def _llm_call_with_retry(client, model: str, messages: list[dict], temperature: float, max_retries: int = 3) -> str:
    """带指数退避的 LLM 调用。"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception:
            if attempt >= max_retries - 1:
                raise
            delay = 2 ** attempt
            print(f"   ⚠️ LLM 调用失败，{delay}s 后重试...")
            time.sleep(delay)
    raise RuntimeError("LLM 调用失败")


class ResearchAgent:
    """论文问答 Agent，封装 RAGSystem 并实现自主规划-执行-反思循环。"""

    def __init__(self, pdf_path: str, max_retries: int = 2):
        self.rag = RAGSystem(pdf_path=pdf_path)
        self.max_retries = max_retries

    # ==================== 初始化 / 清理 ====================

    def initialize(self) -> None:
        """初始化 Milvus + 模型 + PDF 入库。"""
        self.rag.setup_milvus()
        self.rag.setup_models()
        self.rag.ingest_pdf()

    def close(self) -> None:
        self.rag.close()

    # ==================== Agent 主循环 ====================

    def answer(self, query: str) -> dict:
        """Agent 主入口：规划 → 多路检索（含自纠错） → 综合答案。"""
        query = _sanitize(query)
        print(f"\n{'=' * 60}")
        print(f"🤔 问题: {query}")

        # Step 1: 分解 + 策略选择（一次 LLM 调用完成两项）
        plan = self._plan(query)
        if not plan:
            plan = [{"question": query, "strategy": "hybrid"}]

        print(f"📋 分解为 {len(plan)} 个子问题:")
        for i, p in enumerate(plan):
            print(f"   [{i + 1}] ({p['strategy']:>7}) {p['question']}")

        # Step 2: 对每个子问题执行检索 + 自纠错
        search_results: list[dict] = []
        for i, step in enumerate(plan):
            sub_q = step["question"]
            strategy = step["strategy"]
            print(f"\n🔍 [{i + 1}/{len(plan)}] {sub_q[:80]}...")

            results = self._search_with_retry(sub_q, strategy)
            if results:
                search_results.append({
                    "sub_query": sub_q,
                    "strategy": strategy,
                    "results": results,
                })
                print(f"   ✅ {len(results)} 条结果 (最高分 {results[0]['score']:.3f})")
            else:
                print(f"   ❌ 无结果")
                search_results.append({
                    "sub_query": sub_q,
                    "strategy": strategy,
                    "results": [],
                })

        # Step 3: 综合
        if not any(sr["results"] for sr in search_results):
            return {
                "answer": "抱歉，在所有检索策略下均未找到相关信息。建议换个角度提问。",
                "plan": plan,
                "search_results": search_results,
            }

        answer_text = self._synthesize(query, search_results)
        return {
            "answer": answer_text,
            "plan": plan,
            "search_results": search_results,
        }

    # ==================== Step 1: 规划 ====================

    def _plan(self, query: str) -> list[dict]:
        """一次 LLM 调用完成：问题分解 + 检索策略选择。"""
        prompt = (
            "你是一个研究助理。分析以下问题，将其拆解为 1-4 个可独立检索的子问题，"
            "并为每个子问题选择最佳检索策略。\n\n"
            "检索策略说明：\n"
            "- dense: 适用于概念解释、定义、原理描述等语义类问题\n"
            "- keyword: 适用于特定术语、缩写、专有名词、人名等精确匹配\n"
            "- hybrid: 综合问题（不确定时选这个）\n\n"
            "注意：\n"
            "1. 如果原始问题很简单，只返回 1 个子问题即可\n"
            "2. 每个子问题必须独立可检索，不要有「同上」「接着」等依赖\n"
            "3. 只输出 JSON，不要加任何额外文字\n\n"
            f"问题: {query}\n\n"
            '输出格式: {{"sub_queries": [{{"question": "...", "strategy": "dense|keyword|hybrid"}}]}}'
        )

        try:
            raw = _llm_call_with_retry(
                self.rag.llm_client, self.rag.llm_model,
                [{"role": "user", "content": prompt}], 0.3,
            ).strip()
            # 清理可能的 markdown 代码块包裹
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            VALID_STRATEGIES = {"dense", "keyword", "hybrid"}
            data = json.loads(raw)
            sub_queries = data.get("sub_queries", [])
            for sq in sub_queries:
                if sq.get("strategy") not in VALID_STRATEGIES:
                    sq["strategy"] = "hybrid"
            return sub_queries
        except (json.JSONDecodeError, KeyError, Exception):
            print(f"   ⚠️ 规划解析失败，使用原始问题:")
            traceback.print_exc()
            return [{"question": query, "strategy": "hybrid"}]

    # ==================== Step 2: 检索 + 自纠错 ====================

    def _search_with_retry(self, query: str, strategy: str) -> list[dict] | None:
        """带自纠错的检索：结果不够好就改写 query 重试。"""
        current_query = query
        best_results = None

        for attempt in range(self.max_retries + 1):
            # 执行检索
            if strategy == "dense":
                results = self.rag.search_similar(current_query, top_k=5)
            elif strategy == "keyword":
                results = self.rag.keyword_search(current_query, top_k=5)
            else:
                results = self.rag.hybrid_search(current_query, top_k=5)

            results = results or []

            # 保留最佳结果
            if best_results is None or (
                results and results[0]["score"] > (best_results[0]["score"] if best_results else 0)
            ):
                best_results = results

            # 评估是否足够
            if self._evaluate_sufficient(results):
                return results

            # 最后一次不重试
            if attempt >= self.max_retries:
                break

            # 改写 query 重试
            current_query = self._rewrite_query(query, strategy, attempt + 1, results)
            print(f"   🔄 改写: {current_query[:60]}...")

        return best_results if best_results else None

    def _evaluate_sufficient(self, results: list[dict] | None) -> bool:
        """评估检索结果是否足够好。"""
        if not results or len(results) < 2:
            return False
        # 至少有一条分数 > 0.3
        if results[0]["score"] < 0.3:
            return False
        return True

    def _rewrite_query(
        self, original: str, strategy: str, attempt: int, current_results: list[dict] | None
    ) -> str:
        """用 LLM 改写查询以获得更好的检索结果。"""
        hint = ""
        if current_results:
            hint = f"当前共 {len(current_results)} 条结果，但质量偏低。"
        else:
            hint = "当前无任何结果。"

        prompt = (
            f"原始查询：「{original}」\n"
            f"检索策略：{strategy}\n"
            f"{hint}\n\n"
            "请改写此查询以获得更好的检索结果。你可以：\n"
            "- 展开缩写和专业术语\n"
            "- 添加同义词或相关表达\n"
            "- 从不同角度重新表述\n\n"
            "只输出改写后的查询语句，不要加任何额外说明。"
        )

        try:
            return _llm_call_with_retry(
                self.rag.llm_client, self.rag.llm_model,
                [{"role": "user", "content": prompt}], 0.7,
            ).strip().strip('"').strip("'")
        except Exception:
            # LLM 调用失败时做简单规则改写
            traceback.print_exc()
            return f"{original} 详细说明"

    # ==================== Step 3: 综合 ====================

    def _synthesize(self, original_query: str, search_results: list[dict]) -> str:
        """综合所有子问题的检索结果，生成最终答案。"""
        # 构建上下文
        context_parts = []
        for sr in search_results:
            if not sr["results"]:
                continue
            context_parts.append(f"## 子问题: {sr['sub_query']}")
            for i, r in enumerate(sr["results"], 1):
                context_parts.append(
                    f"[来源 页码{r['page']}] {r['text'][:600]}"
                )

        if not context_parts:
            return "未找到相关信息。"

        context = "\n\n".join(context_parts)
        # 截断防止超出 LLM 窗口
        if len(context) > 12000:
            context = context[:12000] + "\n\n[...上下文已截断...]"

        prompt = (
            "你是一个学术论文助手。请基于以下检索到的参考文档内容，"
            "回答用户的原始问题。\n\n"
            "要求：\n"
            "1. 直接回答问题，结构清晰\n"
            "2. 引用具体来源（如 [页码17]）\n"
            "3. 如果某些方面无法从参考文档中找到，请明确说明\n"
            "4. 不要编造参考文档中没有的内容\n\n"
            f"原始问题: {original_query}\n\n"
            f"参考文档:\n{context}\n\n"
            "回答:"
        )

        try:
            return _llm_call_with_retry(
                self.rag.llm_client, self.rag.llm_model,
                [{"role": "user", "content": prompt}], 0.3,
            )
        except Exception:
            traceback.print_exc()
            return "生成答案时出错，请重试。"


# ==================== 交互式入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Agent —— 论文问答智能体")
    parser.add_argument("--query", type=str, default=None, help="单次提问模式")
    parser.add_argument("--pdf", type=str, required=True, help="PDF 文件路径")
    parser.add_argument("--max-retries", type=int, default=2, help="最多重试次数")
    args = parser.parse_args()

    agent = ResearchAgent(pdf_path=args.pdf, max_retries=args.max_retries)

    try:
        agent.initialize()
    except Exception:
        print("❌ 初始化失败:")
        traceback.print_exc()
        agent.close()
        sys.exit(1)

    if args.query:
        # 单次提问模式
        result = agent.answer(args.query)
        print(f"\n{'=' * 60}")
        print("📝 最终答案:")
        print("=" * 60)
        print(result["answer"])
    else:
        # 交互式模式
        print("\n" + "=" * 60)
        print("Research Agent 就绪，输入问题回车即可（输入 /quit 退出）")
        print("=" * 60)

        while True:
            try:
                query = input("\n🔍 请输入查询: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not query:
                continue
            if query.lower() in ("/quit", "/exit", "/q"):
                print("再见！")
                break

            result = agent.answer(query)
            print(f"\n{'=' * 60}")
            print("📝 最终答案:")
            print("=" * 60)
            print(result["answer"])

    agent.close()
    print("资源已释放。")
