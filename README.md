# RAG Paper QA

基于 **Milvus 向量数据库 + Dense/BM25 混合检索 + Reranker 精排** 的学术论文问答系统，并带有一个具备 **问题分解、检索策略选择、自纠错能力** 的 Agent 层。

## 架构

```
用户问题
   │
   ▼
┌──────────────────────────────────────────┐
│  Agent 层 (agent.py)                     │
│  1. 问题分解 (Planning)                  │
│  2. 检索策略选择 (Dense/BM25/Hybrid)     │
│  3. 自纠错 + Query 改写重试              │
│  4. 多子问题结果综合 (Synthesis)         │
└──────────────┬───────────────────────────┘
               │ 调用
               ▼
┌──────────────────────────────────────────┐
│  检索层 (rag/)                           │
│                                          │
│  Query ──→ Dense (BGE + Milvus COSINE)  │
│        ──→ BM25 (本地关键词索引)         │
│        ──→ RRF 加权融合                  │
│        ──→ CrossEncoder Reranker 精排    │
│        ──→ SimHash 去重                  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  生成层                                   │
│  Qwen (阿里百炼) + 来源引用              │
└──────────────────────────────────────────┘
```

## 核心指标

| 指标 | 说明 |
|------|------|
| 检索 | Dense + BM25 + RRF 融合 + Reranker 精排 |
| 去重 | SimHash 指纹 + 汉明距离 ≤3 去重 |
| 分块 | 按页滑动窗口 (500 chars / 150 overlap)，自然句边界截断 |
| 评估 | Recall@K / MRR / NDCG@K，LLM 自动生成测试题 |
| 降级 | Milvus 异常分层捕获；LLM 调用指数退避重试 (最多 3 次) |
| 内存 | 延迟加载 Reranker；GPU FP16 量化；临时 JSONL 流式处理大 PDF |

## 前置依赖

- **Python** ≥ 3.10
- **Milvus** ≥ 2.4（本地或 Docker，见下方一键部署）
- **阿里百炼 API Key**（[免费申请](https://dashscope.console.aliyun.com/)）
- **Tesseract OCR** ≥ 5.x + 中文语言包（仅扫描版 PDF 需要，学术论文通常不需要）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Milvus

```bash
docker-compose up -d
```

等待 10 秒后验证：`docker ps` 应看到 milvus、etcd、minio 三个容器。

### 3. 配置

```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 4. 基础问答

```bash
# 首次运行会解析 PDF、构建索引并入库
python main.py --pdf /path/to/paper.pdf
```

交互式输入问题，输入 `/quit` 退出。

### 5. Agent 模式

```bash
# 单次提问
python agent.py --pdf /path/to/paper.pdf --query "这篇文章的主要贡献是什么？"

# 交互式
python agent.py --pdf /path/to/paper.pdf
```

Agent 会自动拆解复杂问题、选择合适的检索策略、并在结果不佳时改写 query 重试。

### 6. 评估检索质量

```bash
# 自动生成 30 条测试题并计算 Recall / MRR / NDCG
python eval.py --pdf /path/to/paper.pdf

# 对比三种检索策略
python eval.py --pdf /path/to/paper.pdf --compare

# 保存评估结果
python eval.py --pdf /path/to/paper.pdf --output result.json
```

## 配置说明

`.env` 文件：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DASHSCOPE_API_KEY` | 阿里百炼 API Key | — |
| `LLM_BASE_URL` | LLM API 地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_MODEL` | 模型名称 | `qwen-plus` |
| `MILVUS_HOST` | Milvus 地址 | `localhost` |
| `MILVUS_PORT` | Milvus 端口 | `19530` |

## 项目结构

```
.
├── main.py              # 基础 RAG 交互入口
├── agent.py             # Research Agent（规划-检索-自纠错-综合）
├── eval.py              # 检索质量评估（Recall/MRR/NDCG）
├── rag/                 # 核心模块
│   ├── __init__.py
│   ├── rag_system.py    # RAGSystem 类（Milvus/Embedding/BM25/Reranker/LLM）
│   └── utils/
│       ├── config.py    # .env 加载 + Tesseract 配置
│       ├── dedup.py     # SimHash 去重
│       ├── filters.py   # 参考文献/噪声过滤
│       └── coordinates.py  # PDF 坐标提取
├── docker-compose.yml   # Milvus 一键部署
├── requirements.txt
├── .env.example
└── .gitignore
```

## 关键工程细节

- **BM25 索引**：本地 JSON 缓存（`{pdf}.bm25.json`），不依赖 Milvus 读取，重启不丢失
- **Reranker 延迟加载**：入库阶段只加载 Embedding 模型 (≈200MB)，检索时才加载 Reranker (≈2GB)，降低内存峰值
- **GPU FP16**：自动检测 CUDA，Embedding 模型 FP16 量化，内存减半
- **大 PDF 处理**：写入临时 JSONL 文件流式过滤，而非全量驻留内存
- **自然句边界分块**：优先级 — 段落换行 > 中文句号 > 英文句号 > 中文分号 > 空格，避免专业名词被硬切断
- **页眉/页脚过滤**：基于 PDF 坐标识别（出现 ≥3 页的重复文本自动过滤）
- **参考文献/数据噪声过滤**：`[数字]` 引用条目、数字符号占比 >55% 的数据块自动过滤
