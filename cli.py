"""AgentRAG CLI — 自驱式 RAG 框架命令行入口。

用法:
    python cli.py serve                     # 启动 API 服务器
    python cli.py chat --pdf doc.pdf        # 交互式 Agent 对话
    python cli.py ingest --pdf doc.pdf      # 入库文档（可多次调用入库多个文档）
    python cli.py evaluate --pdf doc.pdf    # 检索质量评估
"""

from __future__ import annotations

import asyncio
import sys
import traceback

import typer

app = typer.Typer(
    name="agent-rag",
    help="AgentRAG — 自驱式 RAG 框架，Agent 自主决策检索策略",
)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="绑定地址"),
    port: int = typer.Option(8000, help="绑定端口"),
    reload: bool = typer.Option(False, help="开发模式热重载"),
):
    """启动 FastAPI 服务器。"""
    import uvicorn
    typer.echo(f"Starting RAG Agent API on http://{host}:{port}")
    uvicorn.run(
        "src.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


@app.command()
def chat(
    pdf: str = typer.Option(..., help="PDF 文件路径"),
    query: str | None = typer.Option(None, help="单次提问（不指定则进入交互模式）"),
    max_iterations: int = typer.Option(10, help="Agent 最大迭代次数"),
):
    """启动交互式 Agent 对话（自驱式 RAG 模式）。"""
    from rag import RAGSystem
    from src.agent.agent import ResearchAgent, TokenEvent, ToolCallEvent, ToolResultEvent, DoneEvent

    rag = RAGSystem(pdf_path=pdf)
    agent = ResearchAgent(rag, max_iterations=max_iterations)

    try:
        typer.echo("Initializing...")
        rag.setup_milvus()
        rag.setup_models()
        rag.ingest_pdf()
        agent.initialize()
    except Exception:
        typer.echo("❌ 初始化失败:")
        traceback.print_exc()
        agent.close()
        raise typer.Exit(code=1)

    async def run_query(q: str):
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"🤔 {q}")
        typer.echo(f"{'=' * 60}\n")

        async for event in agent.chat(q):
            match event:
                case ToolCallEvent(tool_name=t, arguments=a):
                    typer.echo(f"  🔧 {t}({a})")
                case ToolResultEvent(tool_name=t, result=r):
                    if r.success:
                        typer.echo(f"  ✅ {t}: {r.metadata.get('count', 0)} results")
                    else:
                        typer.echo(f"  ❌ {t}: {r.error}")
                case TokenEvent(token=tok):
                    typer.echo(tok, nl=False)
                case DoneEvent(final_answer=ans):
                    typer.echo(f"\n\n{'=' * 60}")
                    typer.echo(ans)
                    typer.echo(f"{'=' * 60}")

    if query:
        asyncio.run(run_query(query))
    else:
        typer.echo("\nResearch Agent 就绪，输入问题回车即可 (输入 /quit 退出)")
        typer.echo("=" * 60)

        while True:
            try:
                q = input("\n🔍 ").strip()
            except (EOFError, KeyboardInterrupt):
                typer.echo("\nGoodbye!")
                break
            if not q:
                continue
            if q.lower() in ("/quit", "/exit", "/q"):
                typer.echo("Goodbye!")
                break
            asyncio.run(run_query(q))

    agent.close()
    typer.echo("Resources released.")


@app.command()
def ingest(
    pdf: str = typer.Option(..., help="PDF 文件路径"),
):
    """将 PDF 入库到 Milvus（不启动对话）。"""
    from rag import RAGSystem

    rag = RAGSystem(pdf_path=pdf)
    try:
        typer.echo("Connecting to Milvus...")
        rag.setup_milvus()
        typer.echo("Loading models...")
        rag.setup_models()
        typer.echo("Ingesting PDF...")
        rag.ingest_pdf()
        typer.echo("✅ Ingestion complete.")
    except Exception:
        typer.echo("❌ Ingestion failed:")
        traceback.print_exc()
        raise typer.Exit(code=1)
    finally:
        rag.close()


@app.command()
def evaluate(
    pdf: str = typer.Option(..., help="PDF 文件路径"),
    num: int = typer.Option(30, help="测试问题数量"),
    compare: bool = typer.Option(False, help="对比所有检索策略"),
    seed: int = typer.Option(42, help="随机种子"),
    output: str | None = typer.Option(None, help="JSON 输出路径"),
):
    """运行检索质量评估。"""
    from rag import RAGSystem
    from src.evaluation.metrics import generate_questions, compute_metrics, print_metrics

    rag = RAGSystem(pdf_path=pdf)
    try:
        rag.setup_milvus()
        rag.setup_models()
        rag.ingest_pdf()
        chunks = rag.get_chunks()
        if len(chunks) < num:
            typer.echo(f"Warning: only {len(chunks)} chunks available (requested {num})")
            num = len(chunks)

        typer.echo(f"Generating {num} test questions...")
        test_cases = generate_questions(rag, chunks, num, seed)
        typer.echo(f"Generated {len(test_cases)} questions.\n")

        if compare:
            for strategy, search_fn in [
                ("dense", rag.search_similar),
                ("keyword", rag.keyword_search),
                ("hybrid", rag.hybrid_search),
            ]:
                metrics = compute_metrics(test_cases, search_fn, top_k=5)
                print_metrics(strategy.upper(), metrics)
        else:
            metrics = compute_metrics(test_cases, rag.hybrid_search, top_k=5)
            print_metrics("HYBRID", metrics)

        if output:
            import json
            with open(output, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            typer.echo(f"\nResults saved to {output}")

    except Exception:
        typer.echo("❌ Evaluation failed:")
        traceback.print_exc()
        raise typer.Exit(code=1)
    finally:
        rag.close()


if __name__ == "__main__":
    app()
