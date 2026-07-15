# AgentRAG

<div align="center">

**自驱式 RAG 框架 — Agent 自主决策检索策略，告别被动管道式 RAG**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/suyuan1ya/agent-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/suyuan1ya/agent-rag/actions)
[![Ruff](https://img.shields.io/badge/Lint-Ruff-261230.svg)](https://github.com/astral-sh/ruff)

</div>

## 为什么选择 AgentRAG？

传统 RAG 是**被动的管道**：`Query → Retrieve → Generate`，一步到位，没有反馈回路。

AgentRAG 是**主动的智能体**：

```
用户问题
    │
    ▼
┌─────────────────────────────────────────────────┐
│           AgentRAG 自驱式 Agent 层               │
│                                                  │
│  ① 分析问题意图                                   │
│  ② 选择最优检索策略 (Dense/BM25/Hybrid)            │
│  ③ 执行检索 → 结果评估                            │
│  ④ 不足? → 改写查询 / 拆解问题 → 重试              │
│  ⑤ 足够? → 交叉验证 → 综合输出                      │
│                                                  │
│  ▲ 自主决策  ▲ 自纠错  ▲ 多策略融合  ▲ 有反馈回路    │
└──────────────┬──────────────────────────────────┘
               │ 调用
               ▼
┌──────────────────────────────────────────────────┐
│              混合检索引擎 (rag/)                   │
│                                                  │
│  Dense (BGE + ChromaDB COSINE)                     │
│    + BM25 (本地 TF-IDF 索引)                      │
│    → RRF 加权融合                                 │
│    → CrossEncoder Reranker 精排                   │
│    → SimHash 去重                                 │
└──────────────┬───────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│           LLM 生成 + 来源引用                      │
└──────────────────────────────────────────────────┘
```

## 核心特性

| 特性 | 说明 |
|------|------|
| **自驱式 Agent** | ReAct 模式，LLM 自主选择工具、评估结果、自纠错 |
| **8 个内置工具** | 3 检索 + 3 查询变换 + 2 反思验证，可扩展 |
| **混合检索** | Dense + BM25 + RRF 融合 + CrossEncoder 精排 |
| **SimHash 去重** | 64位指纹 + 汉明距离 ≤3 去重 |
| **检索质量评估** | LLM 自动生成测试题，Recall@K / MRR / NDCG@K |
| **RAGAS 评估** | Faithfulness / Answer Relevancy / Context Precision |
| **LLM-as-Judge** | 准确性、完整性、引用质量、简洁性四维评分 |
| **流式输出** | SSE (Server-Sent Events)，实时展示 Agent 推理过程 |
| **可观测性** | Structlog 结构化日志 + Prometheus 指标 + OpenTelemetry 追踪 |
| **Docker 部署** | 一键 `docker-compose up -d` |

## 技术栈

| 层级 | 技术选型 |
|------|----------|
| Agent 框架 | ReAct + Tool Registry + Event-Driven |
| 向量检索 | ChromaDB (嵌入式) + BGE Embedding (bge-base-zh-v1.5) |
| 关键词检索 | BM25 (k1=1.5, b=0.75) 自实现 + JSON 缓存 |
| 精排 | CrossEncoder Reranker (bge-reranker-v2-m3) |
| LLM | OpenAI 兼容 API（DeepSeek / 通义千问 / OpenAI） |
| API 服务 | FastAPI + SSE Streaming + CORS |
| 文档解析 | PyMuPDF（文字型 PDF） |
| 可观测性 | Structlog + Prometheus + OpenTelemetry |
| 工程化 | Docker / GitHub Actions CI / Pre-commit / Makefile |

## Agent 工具矩阵

### 检索工具 (Search Tools)
| 工具 | 适用场景 |
|------|----------|
| `dense_search` | 语义理解、概念解释、原理描述 |
| `keyword_search` | 精确术语、专有名词、缩写匹配 |
| `hybrid_search` | 综合查询（默认策略，经过 Reranker 精排） |

### 查询变换工具 (Query Tools)
| 工具 | 功能 |
|------|------|
| `decompose_question` | 拆解复杂问题为独立子问题 |
| `rewrite_query` | 改写查询以获得更好检索结果 |
| `generate_hypothetical_answer` | HyDE 策略：生成假设答案用作检索查询 |

### 反思验证工具 (Reflection Tools)
| 工具 | 功能 |
|------|------|
| `evaluate_sufficiency` | 评估检索结果是否充分 |
| `verify_citation` | 验证陈述是否有来源支撑 |

## 快速开始

### 前置依赖

- **Python** >= 3.9
- **LLM API Key**（任一 OpenAI 兼容服务）
- Docker + Docker Compose（可选）

### 1. 克隆项目

```bash
git clone https://github.com/suyuan1ya/agent-rag.git
cd agent-rag
```

### 2. 配置环境

```bash
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
pip install -e ".[dev]"
```

### 3. 启动网页工作台

```bash
python cli.py serve
# 浏览器访问 http://localhost:8000
```

也可以使用 `docker-compose up -d` 启动容器版本。

### 4. 导入文档

推荐直接在网页中选择知识库并上传 PDF。也可以使用 CLI：

```bash
# 可多次执行，入库多个文档到同一知识库
python cli.py ingest --pdf docs/paper1.pdf
python cli.py ingest --pdf docs/paper2.pdf
```

### 5. 使用 Agent

```bash
# 交互式 Agent 对话
python cli.py chat --pdf docs/paper.pdf

# 查询示例：
🔍 这篇文章提出了什么新方法？与传统方法有什么区别？
```

Agent 会自主：
1. 用 `hybrid_search` 检索
2. 调用 `evaluate_sufficiency` 评估结果
3. 如果不够 → `rewrite_query` 改写重试，或 `decompose_question` 拆解
4. 综合输出带引用的答案

### 6. API 服务

```bash
# 启动 API 服务
python cli.py serve

# 网页工作台
open http://localhost:8000

# Swagger API 文档
open http://localhost:8000/docs

# 健康检查
curl http://localhost:8000/api/v1/health

# SSE 流式对话
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"核心贡献是什么？","stream":true}'
```

### 7. 评估检索质量

```bash
# 自动生成测试题并评估
python cli.py evaluate --pdf docs/paper.pdf --num 30

# 对比三种检索策略
python cli.py evaluate --pdf docs/paper.pdf --compare

# 导出评估报告
python cli.py evaluate --pdf docs/paper.pdf --output report.json
```

## 与传统 RAG 的对比

| 维度 | 传统 RAG (LangChain 等) | AgentRAG |
|------|--------------------------|----------|
| 检索策略 | 固定、预配置 | Agent 自主选择 (动态决策) |
| 查询处理 | 一次性 | 多轮改写 + 拆解 + 重试 |
| 反馈回路 | 无 | 检索 → 评估 → 不足 → 重试 |
| 工具使用 | 单一检索工具 | 8 个工具组成的工具箱 |
| 输出质量 | 取决于首次检索 | 自纠错，确保检索充分 |
| 可解释性 | 低（黑盒管道） | 高（每个决策都有事件输出） |

## 项目结构

```
agent-rag/
├── rag/                        # 核心检索引擎
│   ├── rag_system.py           # RAGSystem (ChromaDB / Embedding / BM25 / Reranker)
│   └── utils/                  # 配置、去重、过滤、坐标
├── src/
│   ├── agent/                  # AgentRAG 智能体
│   │   ├── orchestrator.py     # Agent 编排、检索与答案生成
│   │   ├── fsm.py              # 有限状态机与合法状态迁移
│   │   ├── llm.py              # LLM Provider (重试/流式)
│   │   ├── tools/              # 8 个工具 (检索/变换/反思)
│   │   ├── memory/             # 对话记忆 + 工作记忆
│   │   └── prompts/            # 系统提示词
│   ├── api/                    # FastAPI 服务
│   │   ├── app.py              # 应用工厂 + 生命周期
│   │   └── routes/             # chat/search/documents/health
│   ├── infrastructure/rag/     # ChromaDB、BM25 与混合检索
│   ├── web/                    # PDF 上传与问答工作台
│   ├── evaluation/             # 评估体系
│   │   ├── benchmark.py        # 端到端 Benchmark
│   │   ├── judge.py            # LLM-as-Judge
│   │   ├── ragas_eval.py       # RAGAS 指标
│   │   └── metrics.py          # Recall/MRR/NDCG
│   ├── observability/          # 可观测性
│   │   ├── logging.py          # Structlog
│   │   ├── metrics.py          # Prometheus
│   │   └── tracing.py          # OpenTelemetry
│   └── core/
│       └── config.py           # Pydantic Settings 配置
├── tests/                      # 测试
├── cli.py                      # 统一 CLI 入口
├── Dockerfile                  # 应用容器化
├── docker-compose.yml          # Docker 部署
├── Makefile                    # 标准化命令
├── .github/workflows/ci.yml    # CI/CD Pipeline
├── .pre-commit-config.yaml     # Pre-commit Hooks
└── pyproject.toml              # 项目元数据 + 工具配置
```

## 工程亮点

- **BM25 本地索引**：JSON 缓存，独立于向量存储，重启不丢失
- **Reranker 延迟加载**：入库时只加载 Embedding (~200MB)，检索时才加载 Reranker (~2GB)，内存峰值可控
- **GPU FP16 量化**：自动检测 CUDA，Embedding 模型半精度
- **大文件流式处理**：PDF 写入临时 JSONL 过滤，避免全量驻留内存
- **智能分块**：按自然句边界截断（段落 > 句号 > 分号 > 空格），保护专业术语完整性
- **多层降级**：向量存储异常分层捕获，LLM 调用指数退避重试 (3次)
- **事件驱动架构**：Agent 每个动作通过类型化事件通知调用方
- **知识库自动恢复**：服务重启后从 ChromaDB 与 BM25 缓存恢复，无需重复上传
- **本地模型优先**：Embedding 与 Reranker 优先使用 Hugging Face 本地快照

## 常用命令

```bash
make install      # 安装依赖
make lint         # 代码检查
make format       # 代码格式化
make typecheck    # 类型检查
make test         # 运行测试
make test-cov     # 测试 + 覆盖率
make serve        # 启动 API
make docker-up    # 启动所有服务
make docker-down  # 停止所有服务
make clean        # 清理临时文件
```

## 配置参考

`.env` 文件：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DASHSCOPE_API_KEY` | LLM API Key | — |
| `LLM_BASE_URL` | LLM API 地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_MODEL` | 模型名称 | `qwen-plus` |
| `CHROMA_PATH` | ChromaDB 数据路径 | `./chroma_data` |
| `DATA_PATH` | BM25 和运行数据路径 | `./data` |

## License

MIT

## 多知识库 API

API 以 `knowledge_base_id` 隔离索引、Agent 和对话。上传文档时使用 multipart 字段，
聊天和检索时在 JSON 请求中传入相同的知识库 ID：

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@docs/contract.pdf" \
  -F "knowledge_base_id=contracts"

curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"违约责任是什么？","knowledge_base_id":"contracts"}'
```

上传大小、CORS 来源和请求速率分别由 `UPLOAD_MAX_BYTES`、`CORS_ORIGINS`、
`RATE_LIMIT_PER_MINUTE` 配置。生产部署仍应在网关层配置身份认证和持久化任务队列。
