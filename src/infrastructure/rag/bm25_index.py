"""BM25 sparse retrieval index with JSON persistence and SimHash dedup."""

from __future__ import annotations

import json
import math
import os
import re
import traceback
from collections import Counter


def _simhash(text: str) -> int:
    """Compute a 64-bit SimHash fingerprint via 3-gram tokenization and MD5."""
    import hashlib

    grams = [text[i : i + 3] for i in range(max(len(text) - 2, 0))]
    if not grams:
        return 0
    v = [0] * 64
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8", errors="ignore")).hexdigest(), 16)
        for bit in range(64):
            if h & (1 << bit):
                v[bit] += 1
            else:
                v[bit] -= 1
    result = 0
    for bit in range(64):
        if v[bit] > 0:
            result |= 1 << bit
    return result


def _hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit integers."""
    return bin(a ^ b).count("1")


class BM25Index:
    """Standalone BM25 sparse retrieval index with JSON persistence.

    Each instance handles one document set. For multi-tenant scenarios,
    create one BM25Index per tenant.
    """

    def __init__(
        self,
        cache_path: str | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        simhash_threshold: int = 3,
    ):
        self.cache_path = cache_path
        self.k1 = k1
        self.b = b
        self.simhash_threshold = simhash_threshold

        self._docs: list[tuple[int, int, Counter, str]] = []  # (page, doc_len, tf_counter, text)
        self._idf: dict[str, float] = {}
        self._avg_len: float = 0.0
        self._fingerprints: list[int] = []

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    @property
    def vocab_size(self) -> int:
        return len(self._idf)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+|[一-鿿]", text.lower())

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 index from a list of chunks [{text, page_number}, ...]."""
        self._docs = []
        self._idf = {}
        self._avg_len = 0.0
        self._fingerprints = []

        N = len(chunks)
        if N == 0:
            self._save()
            return

        total_len = 0
        for c in chunks:
            text = c["text"]
            tokens = self._tokenize(text)
            self._docs.append((c["page_number"], len(tokens), Counter(tokens), text))
            self._fingerprints.append(_simhash(text))
            total_len += len(tokens)
        self._avg_len = total_len / N

        df: dict[str, int] = {}
        for _, _, counter, _ in self._docs:
            for token in counter:
                df[token] = df.get(token, 0) + 1

        self._idf = {
            token: max(0.0, math.log((N - count + 0.5) / (count + 0.5) + 1))
            for token, count in df.items()
        }
        self._save()

    def search(
        self,
        query: str,
        top_k: int = 10,
        dedup: bool = False,
    ) -> list[dict]:
        """BM25 keyword search returning [{score, page, text}, ...]."""
        if not self._docs or not self._idf:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        avg_len = self._avg_len or 1.0
        scored: list[tuple[float, int, str]] = []

        for page, doc_len, tf_counter, text in self._docs:
            if doc_len == 0:
                continue
            score = 0.0
            for token in query_tokens:
                idf = self._idf.get(token, 0.0)
                if idf == 0.0:
                    continue
                f = tf_counter.get(token, 0)
                if f == 0:
                    continue
                score += (
                    idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * doc_len / avg_len))
                )
            if score > 0:
                scored.append((score, page, text))

        scored.sort(key=lambda x: x[0], reverse=True)

        if dedup:
            return self._dedup_results(scored, top_k)
        return [{"score": s, "page": p, "text": t} for s, p, t in scored[:top_k]]

    def _dedup_results(
        self, scored: list[tuple[float, int, str]], top_k: int
    ) -> list[dict]:
        """Apply SimHash dedup to search results."""
        final: list[dict] = []
        fingerprints: list[int] = []
        for score, page, text in scored:
            fp = _simhash(text)
            if any(_hamming(fp, f) <= self.simhash_threshold for f in fingerprints):
                continue
            final.append({"score": score, "page": page, "text": text})
            fingerprints.append(fp)
            if len(final) >= top_k:
                break
        return final

    def get_chunks(self) -> list[dict]:
        """Return all indexed chunks as [{text, page_number}, ...]."""
        return [{"text": text, "page_number": page} for page, _, _, text in self._docs]

    def get_simhash_fingerprints(self) -> list[int]:
        return list(self._fingerprints)

    # ── persistence ──────────────────────────────────────────

    def _save(self) -> None:
        if not self.cache_path:
            return
        data = {
            "docs": [
                {
                    "page": page,
                    "doc_len": doc_len,
                    "tokens": dict(counter),
                    "text": text,
                    "simhash": fp,
                }
                for (page, doc_len, counter, text), fp in zip(self._docs, self._fingerprints)
            ],
            "idf": self._idf,
            "avg_len": self._avg_len,
        }
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except OSError:
            traceback.print_exc()

    def load(self) -> bool:
        """Load BM25 index from JSON cache. Returns True on success."""
        if not self.cache_path or not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._docs = [
                (d["page"], d["doc_len"], Counter(d["tokens"]), d["text"])
                for d in data["docs"]
            ]
            self._idf = data.get("idf", {})
            self._avg_len = data.get("avg_len", 0.0)
            self._fingerprints = [d.get("simhash", 0) for d in data["docs"]]
            return True
        except (OSError, json.JSONDecodeError, KeyError):
            traceback.print_exc()
            return False
