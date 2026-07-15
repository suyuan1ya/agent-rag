"""Reranker provider — async-safe CrossEncoder wrapper with OOM protection."""

from __future__ import annotations

import asyncio
import threading
import traceback

import numpy as np

from .embeddings import _find_local_snapshot

_RERANKER_LOAD_LOCK = threading.RLock()


class RerankerProvider:
    """Async-safe CrossEncoder reranker with lazy loading and OOM protection.

    Loaded lazily to avoid memory overhead during ingestion. When OOM occurs,
    callers should catch RerankerOOMError and fall back to raw scores.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        min_score: float = 0.1,
    ):
        self.model_name = model_name
        self._device = device
        self.min_score = min_score
        self._model: "CrossEncoder | None" = None  # noqa: F821
        self._lock = threading.RLock()

    def _detect_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._detect_device()
        return self._device

    def load(self) -> None:
        """Load the reranker model synchronously. Call once during init."""
        with _RERANKER_LOAD_LOCK, self._lock:
            if self._model is not None:
                return
            if self._device is None:
                self._device = self._detect_device()

            try:
                from sentence_transformers import CrossEncoder

                local_snapshot = _find_local_snapshot(
                    self.model_name,
                    required_files=("config.json", "tokenizer_config.json"),
                )
                self._model = CrossEncoder(
                    local_snapshot or self.model_name,
                    device=self._device,
                    local_files_only=local_snapshot is not None,
                )
            except Exception as exc:
                traceback.print_exc()
                raise RuntimeError(
                    f"Failed to load reranker model '{self.model_name}' "
                    f"({type(exc).__name__}: {exc})."
                ) from exc

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        """Release reranker memory. Can be reloaded later via load()."""
        if self._model is not None:
            del self._model
            self._model = None
            import gc
            gc.collect()
            if self._device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    async def rerank(
        self, query: str, candidates: list[tuple[str, dict]], top_k: int = 5
    ) -> list[dict]:
        """Rerank candidate documents against the query.

        Args:
            query: search query
            candidates: list of (text, metadata_dict)
            top_k: number of top results to return

        Returns:
            [{score, page, text, ...metadata}, ...] sorted by score descending

        Raises:
            RerankerOOMError: if CUDA OOM occurs (caller should fall back to raw scores)
        """
        if self._model is None:
            self.load()

        if not candidates:
            return []

        try:
            scores = await asyncio.to_thread(self._predict_scores, query, candidates)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "OOM" in str(e):
                raise RerankerOOMError(str(e)) from e
            raise

        # Merge scores with metadata, filter low scores, sort, take top_k
        merged = []
        for (text, meta), score in zip(candidates, scores):
            score_f = float(score)
            if score_f >= self.min_score:
                merged.append({"score": score_f, "text": text, **meta})

        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    def _predict_scores(self, query: str, candidates: list[tuple[str, dict]]):
        with self._lock:
            try:
                pairs = [[query, text] for text, _ in candidates]
                logits = self._model.predict(
                    pairs,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                return 1.0 / (1.0 + np.exp(-logits))
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "OOM" in str(e):
                    raise RerankerOOMError(str(e)) from e
                raise


class RerankerOOMError(RuntimeError):
    """Raised when reranker runs out of GPU memory."""
