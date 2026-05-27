from __future__ import annotations

"""RAG 检索系统核心模块，封装 Milvus 连接、Embedding、BM25、Reranker 和 LLM。"""

import atexit
import gc
import json
import math
import os
import base64
import re
import sys
import time
import traceback
from collections import Counter as _Counter

import numpy as np

# 加载 .env 配置（必须在读取环境变量之前）
from .utils.config import load_dotenv, configure_tesseract
load_dotenv()

# Tesseract OCR 配置（必须在导入 UnstructuredLoader 之前完成）
configure_tesseract()

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from pymilvus.exceptions import MilvusException
from sentence_transformers import SentenceTransformer
from langchain_unstructured import UnstructuredLoader
from sentence_transformers import CrossEncoder
from openai import OpenAI, APIError, AuthenticationError

from .utils.dedup import _simhash, _hamming
from .utils.filters import _is_reference_header, _is_new_section_header, _is_noise_block
from .utils.coordinates import _get_element_y_range, _get_page_height


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


class RAGSystem:
    """RAG 检索系统，封装 Milvus 连接、Embedding、BM25、Reranker 和 LLM。

    所有状态以实例变量管理，避免全局变量。
    """

    def __init__(
        self,
        pdf_path: str,
        milvus_host: str | None = None,
        milvus_port: str | None = None,
        collection_name: str = "pdf_slices",
        embedding_dim: int = 768,
        bm25_cache_path: str | None = None,
        rrf_dense_weight: float = 0.6,
        rrf_sparse_weight: float = 0.4,
    ):
        if milvus_host is None:
            milvus_host = os.getenv("MILVUS_HOST", "localhost")
        if milvus_port is None:
            milvus_port = os.getenv("MILVUS_PORT", "19530")
        self.pdf_path = pdf_path
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.rrf_dense_weight = rrf_dense_weight
        self.rrf_sparse_weight = rrf_sparse_weight

        # BM25 缓存路径
        if bm25_cache_path is None:
            self.bm25_cache_path = os.path.splitext(pdf_path)[0] + ".bm25.json"
        else:
            self.bm25_cache_path = bm25_cache_path

        # 实例变量
        self.collection: Collection | None = None
        self.embed_model: SentenceTransformer | None = None
        self.reranker: CrossEncoder | None = None

        # BM25 内部状态
        self._bm25_docs: list = []       # [(page, doc_len, Counter, text), ...]
        self._bm25_idf: dict = {}        # token -> IDF
        self._bm25_avg_len: float = 0.0

        # SimHash 指纹（配合 BM25 索引存储，检索时用于去重）
        self._simhash_fingerprints: list[int] = []

        # LLM 客户端
        self.llm_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.llm_model = os.getenv("LLM_MODEL", "qwen-plus")

        # 配置
        self.QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
        self.BATCH_SIZE = 100           # 向量插入批量大小
        self.BATCH_ENCODE_SIZE = 64     # 批量 encode 批次大小
        self.MAX_CONTEXT_CHARS = 8000   # LLM 上下文最大字符数
        self.MAX_RESULTS_IN_CONTEXT = 5 # 拼接上下文时的最大结果数
        self.SIMHASH_THRESHOLD = 3      # SimHash 汉明距离阈值（≤3 视为近似重复）
        self.MIN_SCORE = 0.1             # 检索结果最低分数阈值（sigmoid 归一化后，< 0.1 直接丢弃）
        self.VL_MODEL = os.getenv("VL_MODEL", "qwen-vl-plus")  # 图片描述视觉模型
        self.MIN_IMAGE_BYTES = 2000    # 跳过小于 2KB 的图片（图标/装饰元素）
        self.MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 跳过大于 5MB 的图片（API 限制）
        self.MAX_IMAGES_PER_PAGE = 3   # 每页最多处理的图片数
        self.MAX_TOTAL_IMAGES = 30     # 全文档最多处理的图片数

        # 注册退出清理
        atexit.register(self.close)

    # ==================== 清理 ====================

    def close(self) -> None:
        """释放 Milvus 连接、模型和缓存，在退出时自动调用。

        【修复17】同时释放 embedding 模型和 reranker，防止多次运行后内存泄漏。
        """
        # 释放 Milvus 资源
        try:
            if self.collection is not None:
                try:
                    self.collection.release()
                except Exception:
                    pass
                self.collection = None
        except Exception:
            pass
        try:
            connections.disconnect("default")
        except Exception:
            pass

        # 释放模型（GPU 显存 + CPU 内存）
        for attr, name in [("embed_model", "Embedding 模型"), ("reranker", "Reranker")]:
            model = getattr(self, attr, None)
            if model is not None:
                try:
                    del model
                except Exception:
                    pass
                setattr(self, attr, None)
                if hasattr(self, f"_{name}"):
                    continue

        # 强制垃圾回收
        gc.collect()
        if self._device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ==================== 分词 ====================

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'[a-zA-Z0-9]+|[一-鿿]', text.lower())

    # ==================== BM25 索引（本地 JSON 持久化）====================

    def _build_bm25(self, chunks: list[dict]) -> None:
        """基于 chunk 列表构建 BM25 索引和 SimHash 指纹，保存到本地 JSON。"""
        self._bm25_docs = []
        self._bm25_idf = {}
        self._bm25_avg_len = 0.0
        self._simhash_fingerprints = []

        N = len(chunks)
        if N == 0:
            self._save_bm25_cache()
            return

        total_len = 0
        for c in chunks:
            text = c['text']
            tokens = self._tokenize(text)
            self._bm25_docs.append((c['page_number'], len(tokens), _Counter(tokens), text))
            # 【修复8】同步计算 SimHash 指纹
            self._simhash_fingerprints.append(_simhash(text))
            total_len += len(tokens)
        self._bm25_avg_len = total_len / N

        df: dict[str, int] = {}
        for _, _, counter, _ in self._bm25_docs:
            for token in counter:
                df[token] = df.get(token, 0) + 1

        self._bm25_idf = {
            token: max(0.0, math.log((N - count + 0.5) / (count + 0.5) + 1))
            for token, count in df.items()
        }
        self._save_bm25_cache()

    def _save_bm25_cache(self) -> None:
        """将 BM25 索引和 SimHash 指纹保存到本地 JSON。"""
        data = {
            "docs": [
                {
                    "page": page,
                    "doc_len": doc_len,
                    "tokens": dict(counter),
                    "text": text,
                    "simhash": fp,
                }
                for (page, doc_len, counter, text), fp in zip(
                    self._bm25_docs, self._simhash_fingerprints
                )
            ],
            "idf": self._bm25_idf,
            "avg_len": self._bm25_avg_len,
        }
        try:
            os.makedirs(os.path.dirname(self.bm25_cache_path) or ".", exist_ok=True)
            with open(self.bm25_cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except OSError:
            print(f"⚠️ 保存 BM25 缓存失败:")
            traceback.print_exc()

    def _load_bm25_cache(self) -> bool:
        """从本地 JSON 文件加载 BM25 索引和 SimHash 指纹。成功返回 True。"""
        if not os.path.exists(self.bm25_cache_path):
            return False
        try:
            with open(self.bm25_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._bm25_docs = [
                (d["page"], d["doc_len"], _Counter(d["tokens"]), d["text"])
                for d in data["docs"]
            ]
            self._bm25_idf = data.get("idf", {})
            self._bm25_avg_len = data.get("avg_len", 0.0)
            # 【修复8】加载 SimHash 指纹
            self._simhash_fingerprints = [
                d.get("simhash", 0) for d in data["docs"]
            ]
            return True
        except (OSError, json.JSONDecodeError, KeyError):
            print(f"⚠️ BM25 缓存加载失败，将重建:")
            traceback.print_exc()
            return False

    # ==================== Milvus 连接 ====================

    def setup_milvus(self) -> None:
        """连接 Milvus，创建/校验集合。"""
        try:
            connections.connect(alias="default", host=self.milvus_host, port=self.milvus_port)
        except MilvusException:
            print(f"❌ 无法连接 Milvus ({self.milvus_host}:{self.milvus_port}):")
            traceback.print_exc()
            sys.exit(1)

        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                for field in self.collection.schema.fields:
                    if field.name == "embedding":
                        if field.params.get("dim") != self.embedding_dim:
                            print(f"⚠️ 集合维度不匹配，删除重建...")
                            self.collection.drop()
                            self.collection = None
                        break

                if self.collection is not None:
                    self.collection.load()
                    print(f"✅ 已连接已有集合: {self.collection_name}（{self.collection.num_entities} 条数据）")
                    # 【修复2】只从本地缓存加载 BM25，绝不从 Milvus 全量读取
                    if not self._load_bm25_cache():
                        print("⚠️ BM25 缓存缺失，将在入库时重建")
                    return

            # 新建集合
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                FieldSchema(name="page_content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
            ]
            schema = CollectionSchema(fields, description="PDF slices with embeddings")
            self.collection = Collection(self.collection_name, schema, using="default")
            index_params = {"metric_type": "COSINE", "index_type": "FLAT", "params": {}}
            self.collection.create_index(field_name="embedding", index_params=index_params)
            self.collection.load()
            print(f"✅ 已创建新集合: {self.collection_name}")
        except MilvusException:
            print(f"❌ Milvus 操作失败:")
            traceback.print_exc()
            sys.exit(1)

    # 【修复2】删除 _rebuild_bm25_from_milvus —— BM25 仅从本地 JSON 加载，不从 Milvus 全量读取

    # ==================== 模型加载 ====================

    def setup_models(self, load_reranker: bool = False) -> None:
        """加载 embedding 模型（可选 reranker）。

        【修复14】Reranker 默认延迟加载，只在检索阶段才占用内存。
        入库阶段仅需 embedding 模型，可大幅降低内存峰值。
        """
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🖥️  检测到 GPU: {gpu_name}，使用 FP16 加速")
            else:
                print("🖥️  未检测到 GPU，使用 CPU 运行（速度较慢）")
        except ImportError:
            self._device = "cpu"
            print("🖥️  未安装 PyTorch，使用 CPU 运行")

        try:
            self.embed_model = SentenceTransformer(
                'BAAI/bge-base-zh-v1.5',
                device=self._device,
            )
            # 【修复15】GPU 上使用 FP16 量化，内存减半（~400MB → ~200MB）
            if self._device == "cuda":
                self.embed_model.half()
                print("🖥️  Embedding 模型已转为 FP16")
        except Exception:
            print(f"❌ Embedding 模型加载失败:")
            traceback.print_exc()
            sys.exit(1)

        if load_reranker:
            self._load_reranker()

        # CPU 模式下减小批次大小，降低内存峰值
        if self._device == "cpu":
            self.BATCH_ENCODE_SIZE = min(self.BATCH_ENCODE_SIZE, 16)

        loaded = "Embedding 模型"
        if self.reranker is not None:
            loaded += " + Reranker"
        print(f"✅ {loaded} 已加载（device={self._device}）")

    def _load_reranker(self) -> None:
        """延迟加载 reranker，避免入库阶段占用额外内存。"""
        if self.reranker is not None:
            return
        # 【兼容】用 CrossEncoder 替代 FlagReranker，避免 transformers v5 API 不兼容
        try:
            self.reranker = CrossEncoder(
                'BAAI/bge-reranker-v2-m3',
                device=self._device,
            )
            print("✅ Reranker 已加载")
        except Exception:
            print(f"❌ Reranker 模型加载失败:")
            traceback.print_exc()
            sys.exit(1)

    # ==================== PDF 入库 ====================

    def _check_source_exists(self, source_file: str) -> bool:
        """【修复1】检查 source_file 是否已入库，区分"不存在"和"查询错误"。"""
        if self.collection is None or self.collection.num_entities == 0:
            return False
        try:
            # 【修复19】Milvus 表达式将 \ 视为转义符，需转义 Windows 路径
            escaped = source_file.replace('\\', '\\\\')
            results = self.collection.query(
                expr=f'source_file == "{escaped}"',
                output_fields=["id"],
                limit=1,
            )
            return len(results) > 0
        except MilvusException:
            # Milvus 查询异常时打印堆栈，向上抛出而非静默跳过
            print(f"❌ 检查 source_file 时 Milvus 查询异常:")
            traceback.print_exc()
            raise

    def ingest_pdf(self, pdf_path: str | None = None, enable_image_descriptions: bool = True) -> None:
        """解析 PDF、清洗、分块、批量向量化并存入 Milvus。"""
        if pdf_path is None:
            pdf_path = self.pdf_path

        if not os.path.exists(pdf_path):
            print(f"❌ PDF 文件不存在: {pdf_path}")
            sys.exit(1)

        # 【修复1】按 source_file 判断，异常时抛出而非静默跳过
        try:
            if self._check_source_exists(pdf_path):
                print(f"⚠️ PDF 已入库（source_file 校验），跳过。如需重新入库请先手动删除。")
                return
        except MilvusException:
            print("❌ 无法校验入库状态，终止入库以避免重复数据")
            sys.exit(1)

        # 【修复6】本地解析：fast 策略直接从 PDF 提取文本，不触发 OCR
        # 学术论文通常是数字 PDF（非扫描件），fast 速度快、内存低
        try:
            loader = UnstructuredLoader(
                file_path=pdf_path,
                strategy='fast',
                partition_via_api=False,
                coordinates=True,
                languages=['chi_sim'],
            )
        except Exception:
            print(f"❌ PDF Loader 初始化失败:")
            traceback.print_exc()
            sys.exit(1)

        # 【修复18】写入临时 JSONL 文件而非内存列表，避免大 PDF 撑爆 RAM
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.jsonl', prefix='rag_raw_', text=True)
        page_header_texts: dict[int, list[str]] = {}
        page_footer_texts: dict[int, list[str]] = {}
        header_threshold_ratio = 0.08
        footer_threshold_ratio = 0.08
        doc_count = 0
        element_count = 0

        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as tmp:
                for doc in loader.lazy_load():
                    doc_count += 1
                    text = self._clean_text(doc.page_content)
                    if not text:
                        continue
                    page_num = doc.metadata.get("page_number", 0)
                    tmp.write(json.dumps({"t": text, "p": page_num}, ensure_ascii=False) + '\n')
                    element_count += 1

                    # 【修复10】用坐标判断页眉/页脚候选
                    y_range = _get_element_y_range(doc)
                    page_h = _get_page_height(doc)
                    if y_range is not None and page_h is not None:
                        top_y, bottom_y = y_range
                        if top_y < page_h * header_threshold_ratio:
                            page_header_texts.setdefault(page_num, []).append(text)
                        if bottom_y > page_h * (1 - footer_threshold_ratio):
                            page_footer_texts.setdefault(page_num, []).append(text)

                    if doc_count % 50 == 0:
                        print(f"  已解析 {doc_count} 个元素，当前有效切片 {element_count}...")
        except Exception:
            os.unlink(tmp_path)
            print(f"❌ PDF 解析失败:")
            traceback.print_exc()
            print("   请确认 PDF 文件未损坏，或安装本地依赖: pip install 'unstructured[local-inference]'")
            sys.exit(1)
        print(f"🧹 清洗后剩余 {element_count} 个切片")

        if element_count == 0:
            os.unlink(tmp_path)
            print("❌ 清洗后无有效文本")
            sys.exit(1)

        # 【修复10】基于坐标识别页眉/页脚：出现在 >= 3 页的页眉/页脚区域的相同文本
        def _collect_repeating(texts_by_page: dict[int, list[str]]) -> set[str]:
            freq = _Counter()
            for texts in texts_by_page.values():
                freq.update(set(texts))  # 每页每文本只计一次
            return {t for t, c in freq.items() if c >= 3}

        header_texts = _collect_repeating(page_header_texts)
        footer_texts = _collect_repeating(page_footer_texts)
        header_footer = header_texts | footer_texts
        del page_header_texts, page_footer_texts
        print(f"🔍 坐标识别到 {len(header_texts)} 条页眉、{len(footer_texts)} 条页脚")

        # 【修复18】从临时文件流式读取并过滤，不再持有全量 raw_slices 列表
        MIN_TEXT_LEN = 50
        filtered_slices: list[dict] = []
        filtered_header = filtered_short = filtered_ref = filtered_noise = 0
        in_reference_section = False
        try:
            with open(tmp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    rec = json.loads(line)
                    text = rec["t"]
                    if text in header_footer:
                        filtered_header += 1
                        continue

                    # 【修复20】参考文献 / 实验数据过滤
                    if _is_reference_header(text):
                        in_reference_section = True
                        filtered_ref += 1
                        continue
                    if _is_new_section_header(text):
                        in_reference_section = False
                    if in_reference_section:
                        filtered_ref += 1
                        continue
                    if _is_noise_block(text):
                        filtered_noise += 1
                        continue

                    if len(text) < MIN_TEXT_LEN:
                        filtered_short += 1
                        continue
                    filtered_slices.append({"text": text, "page_number": rec["p"]})
        finally:
            os.unlink(tmp_path)
        print(f"🚮 过滤页眉/页脚 {filtered_header} 条，参考文献 {filtered_ref} 条，数据噪声 {filtered_noise} 条，短碎片 {filtered_short} 条")

        if not filtered_slices:
            print("⚠️ 文本过滤后无有效内容，仅尝试图片提取")

        # 图片提取 + 视觉描述
        if enable_image_descriptions:
            print("🖼️  开始提取 PDF 图片并生成描述...")
            image_slices = self._extract_and_describe_images(pdf_path)
            if image_slices:
                filtered_slices.extend(image_slices)
                print(f"✅ 添加 {len(image_slices)} 条图片描述到索引")
            else:
                print("   (无符合条件的图片)")

        if not filtered_slices:
            print("❌ 过滤后无有效文本（含图片描述）")
            sys.exit(1)

        # 构建 chunk
        chunks = self._build_chunks(filtered_slices, chunk_size=500, overlap=150)
        print(f"📦 生成 {len(chunks)} 个语义块")

        # 构建 BM25 索引 + SimHash（自动保存到 JSON）
        self._build_bm25(chunks)
        print(f"📇 BM25 关键词索引已构建（{len(self._bm25_idf)} 个词条，{len(self._simhash_fingerprints)} 个 SimHash 指纹）")
        del filtered_slices
        gc.collect()

        # 【修复3】批量编码，而非逐条 encode（_batch_encode 自动分批避免 OOM）
        total_inserted = 0
        try:
            for i in range(0, len(chunks), self.BATCH_SIZE):
                batch = chunks[i:i + self.BATCH_SIZE]
                batch_texts = [c['text'] for c in batch]

                embeddings = self._batch_encode(batch_texts)

                entities = [
                    {
                        "embedding": emb,
                        "page_content": batch[j]['text'],
                        "page_number": batch[j]['page_number'],
                        "source_file": pdf_path,
                    }
                    for j, emb in enumerate(embeddings)
                ]
                self.collection.insert(entities)
                total_inserted += len(entities)
                print(f"  已插入 {len(entities)} 条，累计 {total_inserted} 条")
        except MilvusException:
            print(f"❌ 向量插入失败:")
            traceback.print_exc()
            sys.exit(1)

        print(f"✅ 全部插入完成，共 {total_inserted} 条")
        try:
            self.collection.flush()
            self.collection.load()
        except MilvusException:
            print(f"❌ Flush/Load 失败:")
            traceback.print_exc()

    def _batch_encode(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化，内部自动分批避免 CUDA OOM。"""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_ENCODE_SIZE):
            sub = texts[i:i + self.BATCH_ENCODE_SIZE]
            embs = self.embed_model.encode(sub, show_progress_bar=False)
            all_embeddings.extend(emb.tolist() for emb in embs)
        return all_embeddings

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r'(\w+)-\s*\r?\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x20-\x7e一-鿿　-〿＀-￯]', '', text)
        return text.strip()

    @staticmethod
    def _build_chunks(slices: list[dict], chunk_size: int = 500, overlap: int = 150) -> list[dict]:
        """按页分组 + 滑动窗口分块，优先在自然句边界处截断。"""
        pages: dict[int, list[str]] = {}
        for s in slices:
            pages.setdefault(s['page_number'], []).append(s['text'])

        chunks: list[dict] = []
        for page_num, texts in pages.items():
            full_text = ' '.join(texts)
            start = 0
            # 【修复16】安全计数器，防止异常参数导致死循环
            max_iterations = max(len(full_text), 1) + 100
            iterations = 0
            while start < len(full_text):
                iterations += 1
                if iterations > max_iterations:
                    print(f"⚠️ 分块超过最大迭代次数，强制退出 (page={page_num})")
                    break
                raw_end = min(start + chunk_size, len(full_text))
                if raw_end < len(full_text):
                    # 【修复21】按优先级查找最佳截断点，避免专业句子被硬切开
                    window = min(200, chunk_size // 2)
                    search_start = max(start, raw_end - window)
                    search_end = min(len(full_text), raw_end + window)
                    region = full_text[search_start:search_end]
                    best = -1

                    # 优先级 1：段落换行（最强边界）
                    m = re.search(r'\n\s*\n', region)
                    if m and search_start + m.start() < raw_end + 60:
                        best = search_start + m.start()

                    # 优先级 2：中文句号 / 问号 / 感叹号
                    if best == -1:
                        for term in ['。', '！', '？']:
                            pos = region.rfind(term, 0, raw_end - search_start + 40)
                            if pos != -1:
                                # 检查不在括号内（后面不能紧接右括号或引号）
                                after = region[pos + 1:pos + 4] if pos + 1 < len(region) else ''
                                if not after.startswith('）') and not after.startswith('」'):
                                    best = search_start + pos + 1
                                    break

                    # 优先级 3：英文句号 .!? 后跟空格+大写（新句开始）
                    if best == -1:
                        m = re.search(r'[.!?]\s+[A-Z]', region)
                        if m:
                            best = search_start + m.start() + 1

                    # 优先级 4：中文分号（弱边界，只在找不到更好位置时用）
                    if best == -1:
                        pos = region.rfind('；', 0, raw_end - search_start + 40)
                        if pos != -1:
                            best = search_start + pos + 1

                    # 兜底：空格
                    if best == -1:
                        pos = region.rfind(' ', raw_end - search_start - 60, raw_end - search_start + 60)
                        if pos != -1:
                            best = search_start + pos

                    if best != -1 and best > start:
                        raw_end = best

                chunk_text = full_text[start:raw_end].strip()
                if len(chunk_text) >= 50:
                    chunks.append({"text": chunk_text, "page_number": page_num})
                    start = raw_end - overlap
                else:
                    start += (chunk_size - overlap)
        return chunks

    # ==================== 图片提取 + 视觉描述 ====================

    def _extract_and_describe_images(self, pdf_path: str) -> list[dict]:
        """从 PDF 中提取图片，调用 VL 模型生成文字描述。

        Returns:
            list[dict]: [{"text": 图片描述, "page_number": int}, ...]
        """
        import fitz  # PyMuPDF，延迟导入避免非必要依赖报错

        try:
            doc = fitz.open(pdf_path)
        except Exception:
            print(f"⚠️ PyMuPDF 无法打开 PDF，跳过图片提取")
            traceback.print_exc()
            return []

        # 尝试加载 Pillow 用于图片缩放
        try:
            from PIL import Image
            import io as _pillow_io
            _has_pillow = True
        except ImportError:
            _has_pillow = False

        MAX_DIM = 1568  # 图片最长边超过此值则等比缩放

        def _prepare_image(image_bytes: bytes, ext: str) -> str | None:
            """准备图片：必要时缩放，返回 base64 data URI。"""
            if _has_pillow:
                try:
                    img = Image.open(_pillow_io.BytesIO(image_bytes))
                    w, h = img.size
                    if max(w, h) > MAX_DIM:
                        ratio = MAX_DIM / max(w, h)
                        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
                    buf = _pillow_io.BytesIO()
                    img_format = "JPEG" if ext.lower() in ("jpeg", "jpg") else ext.upper()
                    if img_format not in ("JPEG", "PNG", "WEBP"):
                        img_format = "JPEG"
                    img.save(buf, format=img_format, quality=85)
                    image_bytes = buf.getvalue()
                    ext = "jpeg"
                except Exception:
                    pass  # 缩放失败就用原图

            b64 = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/{ext};base64,{b64}"

        image_slices: list[dict] = []
        total_images = 0

        for page_num in range(len(doc)):
            if total_images >= self.MAX_TOTAL_IMAGES:
                break

            page = doc[page_num]
            images = page.get_images(full=True)
            if not images:
                continue

            page_img_count = 0
            for img_info in images:
                if total_images >= self.MAX_TOTAL_IMAGES:
                    break
                if page_img_count >= self.MAX_IMAGES_PER_PAGE:
                    break

                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue

                image_bytes = base_image["image"]
                if len(image_bytes) < self.MIN_IMAGE_BYTES:
                    continue
                if len(image_bytes) > self.MAX_IMAGE_BYTES and not _has_pillow:
                    print(f"      ⚠️ 第{page_num + 1}页图片过大 ({len(image_bytes)} bytes) 且无 Pillow，跳过")
                    continue

                ext = base_image["ext"]
                data_uri = _prepare_image(image_bytes, ext)
                if data_uri is None:
                    continue

                description = self._describe_image(data_uri, page_num + 1)
                if description:
                    # 用明确标签标记，方便检索时区分来源
                    image_slices.append({
                        "text": f"[图片描述 | 第{page_num + 1}页] {description}",
                        "page_number": page_num + 1,
                    })
                    page_img_count += 1
                    total_images += 1
                    print(f"   🖼️  第{page_num + 1}页图片{page_img_count}: {description[:60]}...")

            if page_num % 10 == 0 and total_images > 0:
                print(f"   已处理 {total_images} 张图片...")

        doc.close()
        return image_slices

    def _describe_image(self, data_uri: str, page_num: int) -> str | None:
        """调用 qwen-vl-plus 对单张图片生成学术描述。"""
        prompt = (
            "你是一个学术论文审稿人。请用中文详细描述这张图片的内容，包括："
            "1) 图片类型（示意图/流程图/表格/数据图/照片等）；"
            "2) 图中展示的关键信息和数据；"
            "3) 如果有文字标注，请提取关键文字。"
            "描述应控制在 150 字以内，只输出描述文本，不要加任何前缀或引号。"
        )

        try:
            response = _llm_call_with_retry(
                self.llm_client,
                self.VL_MODEL,
                [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                0.3,
                max_retries=2,
            )
            return response.strip().strip('"').strip("'")
        except Exception:
            print(f"   ⚠️ 第{page_num}页图片描述生成失败")
            traceback.print_exc()
            return None

    # ==================== 检索 ====================

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 关键词检索。"""
        if not self._bm25_docs or not self._bm25_idf:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        k1, b = 1.5, 0.75
        avg_len = self._bm25_avg_len or 1.0
        scored: list[tuple[float, int, str]] = []

        for page, doc_len, tf_counter, text in self._bm25_docs:
            if doc_len == 0:
                continue
            score = 0.0
            for token in query_tokens:
                idf = self._bm25_idf.get(token, 0.0)
                if idf == 0.0:
                    continue
                f = tf_counter.get(token, 0)
                if f == 0:
                    continue
                score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / avg_len))
            if score > 0:
                scored.append((score, page, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"score": s, "page": p, "text": t} for s, p, t in scored[:top_k]]

    def search_similar(self, query_text: str, top_k: int = 3) -> list[dict] | None:
        """Dense 向量粗筛 + reranker 精排，SimHash 去重后返回 top_k。"""
        if self.embed_model is None or self.collection is None:
            return None

        try:
            query_vec = self.embed_model.encode(self.QUERY_PREFIX + query_text).tolist()
            search_params = {"metric_type": "COSINE", "params": {}}
            candidates = self.collection.search(
                data=[query_vec],
                anns_field="embedding",
                param=search_params,
                limit=top_k * 5,
                output_fields=["page_content", "page_number", "source_file"],
            )
        except MilvusException:
            print(f"❌ Milvus 向量搜索失败:")
            traceback.print_exc()
            return None

        all_hits = []
        for hits in candidates:
            for hit in hits:
                all_hits.append(hit)

        if not all_hits:
            return None

        self._load_reranker()
        pairs = [[query_text, hit.entity.get('page_content')] for hit in all_hits]
        logits = self.reranker.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        # sigmoid 归一化到 [0, 1]，CrossEncoder 返回原始 logits
        scores = 1.0 / (1.0 + np.exp(-logits))

        # 过滤低分结果 + 按分数降序排列
        scored = [(hit, s) for hit, s in zip(all_hits, scores) if s >= self.MIN_SCORE]
        if not scored:
            return None
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)

        # 【修复8】SimHash 去重，替换 Jaccard 字符重叠
        final: list[tuple] = []
        final_fingerprints: list[int] = []
        for hit, score in ranked:
            text = hit.entity.get('page_content')
            fp = _simhash(text)
            if any(_hamming(fp, existing) <= self.SIMHASH_THRESHOLD for existing in final_fingerprints):
                continue
            final.append((hit, score))
            final_fingerprints.append(fp)
            if len(final) >= top_k:
                break

        # 【修复4】去重后为空则返回 None，不传给 LLM
        if not final:
            return None

        return [
            {"score": score, "page": hit.entity.get('page_number'), "text": hit.entity.get('page_content')}
            for hit, score in final
        ]

    def hybrid_search(self, query: str, top_k: int = 3) -> list[dict] | None:
        """Dense + BM25 混合检索，加权 RRF 融合后 reranker 精排。

        【修复9】RRF 加入向量/关键词权重，dense 和 sparse 可独立调权。
        """
        RRF_K = 60

        dense_results = self.search_similar(query, top_k=top_k * 5) or []
        sparse_results = self.keyword_search(query, top_k=top_k * 5)

        rrf_scores: dict[tuple, float] = {}
        doc_map: dict[tuple, dict] = {}

        # 【修复9】向量检索用 rrf_dense_weight，关键词检索用 rrf_sparse_weight
        for rank, r in enumerate(dense_results):
            key = (r['page'], r['text'])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.rrf_dense_weight / (RRF_K + rank + 1)
            doc_map[key] = r

        for rank, r in enumerate(sparse_results):
            key = (r['page'], r['text'])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.rrf_sparse_weight / (RRF_K + rank + 1)
            doc_map[key] = r

        if not rrf_scores:
            return None

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 5]
        candidates = [(doc_map[key], score) for key, score in sorted_items]

        self._load_reranker()
        pairs = [[query, c[0]['text']] for c in candidates]
        logits = self.reranker.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        scores = 1.0 / (1.0 + np.exp(-logits))

        # 过滤低分 + 按分数降序
        scored = [(c[0], s) for c, s in zip(candidates, scores) if s >= self.MIN_SCORE]
        if not scored:
            return None
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)

        # 【修复8】SimHash 去重
        final: list[dict] = []
        final_fingerprints: list[int] = []
        for doc, score in ranked:
            fp = _simhash(doc['text'])
            if any(_hamming(fp, existing) <= self.SIMHASH_THRESHOLD for existing in final_fingerprints):
                continue
            final.append({"score": score, "page": doc['page'], "text": doc['text']})
            final_fingerprints.append(fp)
            if len(final) >= top_k:
                break

        # 【修复4】去重后为空则直接返回 None
        if not final:
            return None

        return final

    # ==================== 答案生成 ====================

    def generate_answer(self, query: str, results: list[dict]) -> str:
        """基于检索结果调用 LLM 生成答案。

        【修复5】限制上下文总长度，只取高分前 N 条，硬截断避免超出 LLM 窗口。
        """
        # 按分数降序排列，只取前 MAX_RESULTS_IN_CONTEXT 条
        sorted_results = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
        limited = sorted_results[:self.MAX_RESULTS_IN_CONTEXT]

        # 拼接上下文，硬限制总字符数
        context_parts: list[str] = []
        total_chars = 0
        for i, r in enumerate(limited, 1):
            part = f"[来源{i} | 页码{r['page']}]\n{r['text']}"
            if total_chars + len(part) > self.MAX_CONTEXT_CHARS:
                part = part[:self.MAX_CONTEXT_CHARS - total_chars] + "..."
                context_parts.append(part)
                break
            context_parts.append(part)
            total_chars += len(part)

        context = "\n\n".join(context_parts)

        system_prompt = (
            "你是一个学术论文助手。请严格基于以下提供的参考文档内容回答用户问题。"
            "如果参考文档中没有足够的信息，请明确说明。"
            "在回答中引用来源编号（如 [来源1]）。"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"参考文档：\n\n{context}\n\n问题：{query}"},
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content
        except AuthenticationError:
            print("❌ 千问 API key 无效，请检查 DASHSCOPE_API_KEY 环境变量")
            traceback.print_exc()
            raise
        except APIError:
            print(f"❌ 千问 API 调用失败:")
            traceback.print_exc()
            raise
        except Exception:
            print(f"❌ LLM 调用未知错误:")
            traceback.print_exc()
            raise

    # ==================== BM25 文档暴露（供 eval 使用）====================

    def get_chunks(self) -> list[dict]:
        """返回 BM25 中所有 chunk 的 {text, page_number} 列表。"""
        return [{"text": text, "page_number": page} for page, _, _, text in self._bm25_docs]
