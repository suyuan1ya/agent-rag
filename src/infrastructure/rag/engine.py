"""RAG Engine — composition root that ties together all RAG infrastructure.

This is the single entry point for retrieval operations, replacing the
monolithic RAGSystem with a composed set of modular providers.
"""

from __future__ import annotations

import os
import traceback

from src.infrastructure.fault_tolerance.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)

from .bm25_index import BM25Index, _hamming, _simhash
from .embeddings import EmbeddingProvider
from .ingestion import IngestionPipeline
from .reranker import RerankerProvider
from .vector_store import VectorStore


class RAGEngine:
    """Composed RAG engine with Dense/BM25/Hybrid search strategies.

    Usage:
        engine = RAGEngine(chroma_path="./data", collection_name="contracts")
        engine.initialize()
        await engine.ingest("contract.pdf")
        results = await engine.hybrid_search("违约金条款是什么?", top_k=5)
    """

    def __init__(
        self,
        chroma_path: str | None = None,
        collection_name: str = "documents",
        embedding_dim: int = 768,
        embedding_model: str = "BAAI/bge-base-zh-v1.5",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        rrf_dense_weight: float = 0.6,
        rrf_sparse_weight: float = 0.4,
        simhash_threshold: int = 3,
        min_score: float = 0.1,
        query_prefix: str = "为这个句子生成表示以用于检索相关文章：",
    ):
        if chroma_path is None:
            chroma_path = os.getenv("CHROMA_PATH", "./chroma_data")

        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.rrf_dense_weight = rrf_dense_weight
        self.rrf_sparse_weight = rrf_sparse_weight
        self.simhash_threshold = simhash_threshold
        self.min_score = min_score

        # Providers (created in initialize())
        self.embedding = EmbeddingProvider(
            model_name=embedding_model,
            query_prefix=query_prefix,
        )
        self.vector_store = VectorStore(
            chroma_path=chroma_path,
            collection_name=collection_name,
            embedding_dim=embedding_dim,
        )
        self.bm25 = BM25Index(simhash_threshold=simhash_threshold)
        self.reranker = RerankerProvider(
            model_name=reranker_model,
            min_score=min_score,
        )

        self._initialized = False
        self._degraded_dense = False  # set to True when dense retrieval is unavailable
        self._degraded_reranker = False  # set to True when reranker is unavailable
        self._dense_breaker = CircuitBreaker("dense", failure_threshold=3, recovery_timeout=30)
        self._reranker_breaker = CircuitBreaker(
            "reranker", failure_threshold=2, recovery_timeout=60
        )

    # ── lifecycle ────────────────────────────────────────────

    def initialize(
        self,
        pdf_path: str = "",
        load_reranker: bool = False,
        require_documents: bool = False,
    ) -> bool:
        """Initialize all providers and connect to vector store.

        Args:
            pdf_path: path to PDF (used to derive BM25 cache path)
            load_reranker: if True, load reranker immediately (lazy by default)
        """
        connected = self.vector_store.connect(create_if_missing=not require_documents)
        if not connected:
            return False
        if require_documents and self.vector_store.count_sync() == 0:
            return False
        self.embedding.load()

        if pdf_path:
            self.bm25.cache_path = os.path.splitext(pdf_path)[0] + ".bm25.json"
        if self.bm25.cache_path:
            self.bm25.load()

        if load_reranker:
            self.reranker.load()

        self._initialized = True
        return True

    def close(self) -> None:
        """Release all resources."""
        self.vector_store.close()
        self.embedding.close()
        self.reranker.unload()

    # ── retrieval ────────────────────────────────────────────

    async def dense_search(self, query: str, top_k: int = 5) -> list[dict] | None:
        """Semantic vector search with reranker and SimHash dedup."""
        count = self.vector_store.count_sync()
        if count == 0:
            return None
        try:
            chroma_result = await self._dense_breaker.call(
                self._dense_retrieve, query, min(top_k * 5, count)
            )
        except CircuitOpenError:
            return None
        except Exception:
            traceback.print_exc()
            return None

        doc_texts = chroma_result.get("documents", [[]])[0]
        metadatas = chroma_result.get("metadatas", [[]])[0]
        distances = chroma_result.get("distances", [[]])[0]

        if not doc_texts:
            return None

        # Convert cosine distance to similarity
        candidates = [
            (text, {"page": meta.get("page_number", 0)})
            for text, meta, dist in zip(doc_texts, metadatas, distances)
        ]

        # Rerank (with fallback to raw scores)
        try:
            results = await self._reranker_breaker.call(
                self.reranker.rerank, query, candidates, top_k=top_k * 2
            )
        except Exception:
            results = self._raw_score_results(candidates, distances, top_k * 2)

        # SimHash dedup
        return self._simhash_dedup(results, top_k)

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 keyword search (synchronous, no external API calls)."""
        return self.bm25.search(query, top_k=top_k, dedup=True)

    async def hybrid_search(self, query: str, top_k: int = 5) -> list[dict] | None:
        """Dense + BM25 hybrid search with RRF fusion, reranker, and dedup."""
        RRF_K = 60

        dense_results = await self.dense_search(query, top_k=top_k * 5) or []
        sparse_results = self.keyword_search(query, top_k=top_k * 5)

        if not dense_results and not sparse_results:
            return None

        # RRF fusion with configurable weights
        rrf_scores: dict[tuple, float] = {}
        doc_map: dict[tuple, dict] = {}

        for rank, r in enumerate(dense_results):
            key = (r["page"], r["text"])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.rrf_dense_weight / (RRF_K + rank + 1)
            doc_map[key] = r

        for rank, r in enumerate(sparse_results):
            key = (r["page"], r["text"])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.rrf_sparse_weight / (RRF_K + rank + 1)
            doc_map[key] = r

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: top_k * 5]
        candidates = [
            (doc_map[key]["text"], {"page": doc_map[key]["page"]}) for key, _ in sorted_items
        ]

        # Rerank (with fallback)
        try:
            results = await self._reranker_breaker.call(
                self.reranker.rerank, query, candidates, top_k=top_k * 2
            )
        except Exception:
            results = [
                {"score": score, "text": doc_map[key]["text"], "page": doc_map[key]["page"]}
                for key, score in sorted_items[:top_k]
            ]

        return self._simhash_dedup(results, top_k)

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _raw_score_results(
        candidates: list[tuple[str, dict]],
        distances: list[float],
        top_k: int,
    ) -> list[dict]:
        """Build results from raw cosine distances (no reranker)."""
        results = []
        for (text, meta), dist in zip(candidates, distances):
            score = 1.0 - dist
            results.append({"score": score, "text": text, **meta})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def _dense_retrieve(self, query: str, n_results: int) -> dict:
        query_vec = (await self.embedding.encode(query))[0]
        return await self.vector_store.query(query_vec, n_results=n_results)

    def _simhash_dedup(self, results: list[dict], top_k: int) -> list[dict] | None:
        """Deduplicate results by SimHash fingerprint."""
        final: list[dict] = []
        fingerprints: list[int] = []
        for r in results:
            fp = _simhash(r["text"])
            if any(_hamming(fp, f) <= self.simhash_threshold for f in fingerprints):
                continue
            final.append(r)
            fingerprints.append(fp)
            if len(final) >= top_k:
                break
        return final if final else None

    # ── ingestion ────────────────────────────────────────────

    async def ingest(self, pdf_path: str, source_name: str | None = None) -> int:
        """Ingest a PDF document into the search index."""
        pipeline = IngestionPipeline(
            vector_store=self.vector_store,
            embedding_provider=self.embedding,
            bm25_index=self.bm25,
        )
        return await pipeline.ingest(pdf_path, source_name=source_name)

    # ── chunk access ─────────────────────────────────────────

    def get_chunks(self) -> list[dict]:
        """Return all indexed chunks for evaluation purposes."""
        return self.bm25.get_chunks()

    # ── degradation status ───────────────────────────────────

    @property
    def degradation_status(self) -> dict:
        """Return current degradation state for health checks."""
        return {
            "dense_available": self._dense_breaker.state is not CircuitState.OPEN,
            "reranker_available": self._reranker_breaker.state is not CircuitState.OPEN,
            "bm25_docs": self.bm25.doc_count,
            "vector_docs": self.vector_store.count_sync(),
        }
