"""RAG 交互式查询入口。

用法:
    python main.py                          # 交互式问答
    python main.py --pdf <path>             # 指定 PDF 路径
"""

import argparse
import sys
import traceback

from rag import RAGSystem


def main():
    parser = argparse.ArgumentParser(description="RAG 论文问答系统")
    parser.add_argument("--pdf", type=str, required=True, help="PDF 文件路径")
    args = parser.parse_args()

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

    print("\n" + "=" * 60)
    print("RAG 查询就绪，输入问题回车即可（输入 /quit 退出）")
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

        # 【修复13】异常分类捕获，打印堆栈
        try:
            results = rag.hybrid_search(query)
        except Exception:
            print(f"❌ 检索失败 (Milvus):")
            traceback.print_exc()
            continue

        # 【修复4】防御：结果为空时不调用 LLM
        if not results:
            print("无检索结果，请换个问法试试。")
            continue

        print(f"\n📖 检索到 {len(results)} 条相关内容：")
        for r in results:
            print(f"  [{r['score']:.3f}] p{r['page']} | {r['text'][:100]}...")

        print("\n" + "-" * 40)
        print("🤖 答案：")
        print("-" * 40)
        try:
            from openai import AuthenticationError, APIError
            answer = rag.generate_answer(query, results)
            print(answer)
        except AuthenticationError:
            print("❌ 千问 API key 无效，请检查 DASHSCOPE_API_KEY 环境变量")
        except APIError:
            print(f"❌ 千问 API 调用失败:")
            traceback.print_exc()
        except Exception:
            print(f"❌ 生成答案时出错:")
            traceback.print_exc()

    # 【修复12】主动释放资源
    rag.close()
    print("资源已释放。")


if __name__ == "__main__":
    main()
