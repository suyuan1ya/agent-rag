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
from typing import Optional

# 修复 Windows GBK 终端下的 emoji 打印问题
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

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
    query: Optional[str] = typer.Option(None, help="单次提问（不指定则进入交互模式）"),
    max_iterations: int = typer.Option(15, help="Agent 最大迭代次数"),
):
    """启动交互式 Agent 对话（自驱式 RAG 模式）。"""
    from src.agent.orchestrator import DoneEvent, TokenEvent, ToolCallEvent, ToolResultEvent
    from src.core.config import get_settings
    from src.core.runtime import create_agent, create_engine

    settings = get_settings()
    rag = create_engine("cli", settings)

    try:
        typer.echo("Initializing...")
        rag.initialize()
        asyncio.run(rag.ingest(pdf))
        agent = create_agent(rag, settings)
    except Exception:
        typer.echo("❌ 初始化失败:")
        traceback.print_exc()
        rag.close()
        raise typer.Exit(code=1)

    async def run_query(q: str):
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"🤔 {q}")
        typer.echo(f"{'=' * 60}\n")

        async for event in agent.run(q):
            if isinstance(event, ToolCallEvent):
                typer.echo(f"  🔧 {event.tool_name}({event.arguments})")
            elif isinstance(event, ToolResultEvent):
                if event.result.success:
                    typer.echo(
                        f"  ✅ {event.tool_name}: {event.result.metadata.get('count', 0)} results"
                    )
                else:
                    typer.echo(f"  ❌ {event.tool_name}: {event.result.error}")
            elif isinstance(event, TokenEvent):
                typer.echo(event.token, nl=False)
            elif isinstance(event, DoneEvent):
                typer.echo(f"\n\n{'=' * 60}")
                typer.echo(event.final_answer)
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
    """将 PDF 入库到 ChromaDB（不启动对话）。"""
    from src.core.runtime import create_engine

    rag = create_engine("default")
    try:
        typer.echo("Initializing vector store...")
        rag.initialize()
        typer.echo("Ingesting PDF...")
        asyncio.run(rag.ingest(pdf))
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
    output: Optional[str] = typer.Option(None, help="JSON 输出路径"),
):
    """运行检索质量评估。"""
    from src.core.runtime import create_engine
    from src.evaluation.metrics import compute_metrics, generate_questions, print_metrics

    rag = create_engine("evaluation")
    try:
        rag.initialize()
        asyncio.run(rag.ingest(pdf))
        chunks = rag.get_chunks()
        if len(chunks) < num:
            typer.echo(f"Warning: only {len(chunks)} chunks available (requested {num})")
            num = len(chunks)

        typer.echo(f"Generating {num} test questions...")
        test_cases = generate_questions(rag, chunks, num, seed)
        typer.echo(f"Generated {len(test_cases)} questions.\n")

        if compare:
            for strategy, search_fn in [
                ("dense", lambda q, top_k=5: asyncio.run(rag.dense_search(q, top_k))),
                ("keyword", rag.keyword_search),
                ("hybrid", lambda q, top_k=5: asyncio.run(rag.hybrid_search(q, top_k))),
            ]:
                metrics = compute_metrics(test_cases, search_fn, top_k=5)
                print_metrics(strategy.upper(), metrics)
        else:
            metrics = compute_metrics(
                test_cases,
                lambda q, top_k=5: asyncio.run(rag.hybrid_search(q, top_k)),
                top_k=5,
            )
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
