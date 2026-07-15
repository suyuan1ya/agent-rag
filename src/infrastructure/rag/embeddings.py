"""Embedding provider — async-safe wrapper around SentenceTransformer."""

from __future__ import annotations

import asyncio
import os
import threading
import traceback
from pathlib import Path

_MODEL_LOAD_LOCK = threading.RLock()


def _find_local_snapshot(
    model_name: str,
    required_files: tuple[str, ...] = ("config.json",),
) -> str | None:
    """Resolve a complete cached Hugging Face snapshot without relying on shell env."""
    repo_dir = "models--" + model_name.replace("/", "--")
    roots = [Path.home() / ".cache" / "huggingface" / "hub"]
    if os.getenv("HF_HOME"):
        roots.insert(0, Path(os.environ["HF_HOME"]).expanduser() / "hub")

    for root in roots:
        model_dir = root / repo_dir
        snapshots_dir = model_dir / "snapshots"
        ref_file = model_dir / "refs" / "main"
        candidates: list[Path] = []
        if ref_file.is_file():
            candidates.append(snapshots_dir / ref_file.read_text(encoding="utf-8").strip())
        if snapshots_dir.is_dir():
            candidates.extend(
                sorted(snapshots_dir.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
            )
        for candidate in candidates:
            if all((candidate / filename).is_file() for filename in required_files):
                return str(candidate)
    return None


class EmbeddingProvider:
    """Async-safe embedding model wrapper with GPU/CPU auto-detection.

    Manages a singleton SentenceTransformer instance behind a thread lock so
    the provider can safely move between the API loop and ingestion workers.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        device: str | None = None,
        query_prefix: str = "为这个句子生成表示以用于检索相关文章：",
        batch_encode_size: int = 64,
    ):
        self.model_name = model_name
        self._device = device
        self.query_prefix = query_prefix
        self.batch_encode_size = batch_encode_size
        self._model: "SentenceTransformer | None" = None  # noqa: F821
        self._lock = threading.RLock()

    @property
    def device(self) -> str:
        if self._device is None:
            self._detect_device()
        return self._device

    def _detect_device(self) -> None:
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self._device = "cpu"

    def load(self) -> None:
        """Load the embedding model (synchronous, call once during init)."""
        with _MODEL_LOAD_LOCK, self._lock:
            if self._model is not None:
                return
            if self._device is None:
                self._detect_device()

            try:
                from sentence_transformers import SentenceTransformer

                local_snapshot = _find_local_snapshot(
                    self.model_name,
                    required_files=("config.json", "modules.json"),
                )
                model_source = local_snapshot or self.model_name
                self._model = SentenceTransformer(
                    model_source,
                    device=self._device,
                    local_files_only=local_snapshot is not None,
                )
                if self._device == "cuda":
                    self._model.half()

                if self._device == "cpu":
                    self.batch_encode_size = min(self.batch_encode_size, 16)
            except Exception as exc:
                traceback.print_exc()
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}' "
                    f"({type(exc).__name__}: {exc})."
                ) from exc

    def is_loaded(self) -> bool:
        return self._model is not None

    async def encode(self, texts: str | list[str], add_query_prefix: bool = True) -> list[list[float]]:
        """Encode one or more texts to embedding vectors (async-safe)."""
        if self._model is None:
            raise RuntimeError("Embedding model not loaded. Call load() first.")

        if isinstance(texts, str):
            texts = [texts]
        if add_query_prefix:
            texts = [self.query_prefix + t for t in texts]

        return await asyncio.to_thread(self._encode_batches, texts)

    def encode_sync(self, texts: str | list[str], add_query_prefix: bool = True) -> list[list[float]]:
        """Synchronous encode for non-async contexts."""
        if self._model is None:
            raise RuntimeError("Embedding model not loaded. Call load() first.")

        if isinstance(texts, str):
            texts = [texts]
        if add_query_prefix:
            texts = [self.query_prefix + t for t in texts]

        return self._encode_batches(texts)

    def _encode_batches(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            all_embeddings: list[list[float]] = []
            for i in range(0, len(texts), self.batch_encode_size):
                batch = texts[i : i + self.batch_encode_size]
                embs = self._model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(emb.tolist() for emb in embs)
            return all_embeddings

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
