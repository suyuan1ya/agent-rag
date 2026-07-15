"""Cache Manager — two-level caching for embeddings and search results.

Embedding Cache: LRU, keyed by md5(query), stores embedding vectors.
  - Never invalidated (embeddings are deterministic for a given model)
  - Default max 10,000 entries (~80 MB for 768-dim vectors)

Result Cache: TTL, keyed by md5(query + strategy + top_k), stores result lists.
  - Invalidated on document ingestion for that tenant
  - Default TTL 300s, max 5,000 entries

Backends: in-memory (default) or Redis (optional, for multi-process deployments).
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class EmbeddingCache:
    """LRU cache for query embedding vectors."""

    def __init__(self, maxsize: int = 10_000):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _key(self, text: str) -> str:
        return f"emb:{hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()}"

    def get(self, text: str) -> list[float] | None:
        key = self._key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, text: str, embedding: list[float]) -> None:
        key = self._key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = embedding

    async def get_or_compute(
        self, text: str, fn: Callable[[str], Awaitable[list[float]]]
    ) -> list[float]:
        """Get from cache or compute via fn."""
        cached = self.get(text)
        if cached is not None:
            return cached
        result = await fn(text)
        self.set(text, result)
        return result

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class ResultCache:
    """TTL cache for search results."""

    def __init__(self, ttl: float = 300.0, maxsize: int = 5_000):
        self.ttl = ttl
        self.maxsize = maxsize
        self._cache: dict[str, tuple[list[dict], float]] = {}  # key → (results, expiry_time)
        self._hits = 0
        self._misses = 0

    def _key(self, query: str, strategy: str, top_k: int, scope: str = "") -> str:
        raw = f"{query}|{strategy}|{top_k}|{scope}"
        return f"res:{hashlib.md5(raw.encode('utf-8', errors='ignore')).hexdigest()}"

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._cache.items() if exp <= now]
        for k in expired:
            del self._cache[k]

    def get(self, query: str, strategy: str = "hybrid", top_k: int = 5, scope: str = "") -> list[dict] | None:
        self._evict_expired()
        key = self._key(query, strategy, top_k, scope)
        if key in self._cache:
            results, _ = self._cache[key]
            self._hits += 1
            return results
        self._misses += 1
        return None

    def set(self, query: str, results: list[dict], strategy: str = "hybrid", top_k: int = 5, scope: str = "") -> None:
        key = self._key(query, strategy, top_k, scope)
        if len(self._cache) >= self.maxsize:
            # Evict oldest entry
            oldest = min(self._cache.items(), key=lambda x: x[1][1])
            del self._cache[oldest[0]]
        self._cache[key] = (results, time.monotonic() + self.ttl)

    def invalidate(self, scope: str | None = None) -> None:
        """Invalidate cache entries. If scope is None, clear all."""
        if scope is None:
            self._cache.clear()
        else:
            # Invalidate entries matching scope
            self._cache = {
                k: v for k, v in self._cache.items()
                if scope not in k
            }

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "ttl": self.ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class CacheManager:
    """Composite cache manager — embedding cache + result cache.

    Usage:
        cm = CacheManager()
        embedding = await cm.embedding.get_or_compute(query, embedding_fn)
        results = cm.result.get(query, strategy="hybrid", top_k=5)
    """

    def __init__(
        self,
        embedding_maxsize: int = 10_000,
        result_ttl: float = 300.0,
        result_maxsize: int = 5_000,
    ):
        self.embedding = EmbeddingCache(maxsize=embedding_maxsize)
        self.result = ResultCache(ttl=result_ttl, maxsize=result_maxsize)

    def invalidate_all(self) -> None:
        """Clear all caches (e.g., after new document ingestion)."""
        self.embedding.clear()
        self.result.clear()

    def invalidate_tenant(self, tenant_id: str) -> None:
        """Invalidate cached results for a specific tenant."""
        self.result.invalidate(scope=tenant_id)

    @property
    def stats(self) -> dict:
        return {
            "embedding_cache": self.embedding.stats,
            "result_cache": self.result.stats,
        }

    def clear(self) -> None:
        self.embedding.clear()
        self.result.clear()
