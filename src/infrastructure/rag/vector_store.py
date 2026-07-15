"""Vector store abstraction over ChromaDB with async-safe operations."""

from __future__ import annotations

import asyncio
import threading
import traceback
from typing import Any


class VectorStore:
    """Async-safe wrapper around ChromaDB PersistentClient.

    Operations are serialized through a thread lock so the store can be used
    by both API event loops and background ingestion workers.
    """

    def __init__(
        self,
        chroma_path: str = "./chroma_data",
        collection_name: str = "documents",
        embedding_dim: int = 768,
    ):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._client: Any = None
        self._collection: Any = None
        self._lock = threading.RLock()

    @property
    def collection(self):
        return self._collection

    def connect(self, create_if_missing: bool = True) -> bool:
        """Initialize ChromaDB client and get-or-create collection (sync)."""
        import chromadb

        try:
            self._client = chromadb.PersistentClient(path=self.chroma_path)
        except Exception:
            traceback.print_exc()
            raise RuntimeError(f"Cannot initialize ChromaDB at {self.chroma_path}")

        try:
            if create_if_missing:
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                collection_names = {c.name for c in self._client.list_collections()}
                if self.collection_name not in collection_names:
                    return False
                self._collection = self._client.get_collection(name=self.collection_name)
            count = self._collection.count()
            if count > 0:
                print(f"Connected to existing collection '{self.collection_name}' ({count} docs)")
            else:
                print(f"Created new collection '{self.collection_name}' at {self.chroma_path}")
            return True
        except Exception:
            traceback.print_exc()
            raise RuntimeError(f"ChromaDB operation failed for collection '{self.collection_name}'")

    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Add documents to the collection (async-safe)."""
        if self._collection is None:
            raise RuntimeError("Vector store not connected. Call connect() first.")
        def operation():
            with self._lock:
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )

        await asyncio.to_thread(operation)

    async def query(
        self, query_embedding: list[float], n_results: int = 20, where: dict | None = None
    ) -> dict:
        """Query the collection for similar documents (async-safe)."""
        if self._collection is None:
            raise RuntimeError("Vector store not connected.")
        def operation():
            with self._lock:
                return self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                    where=where,
                )

        return await asyncio.to_thread(operation)

    async def count(self) -> int:
        if self._collection is None:
            return 0
        return await asyncio.to_thread(self.count_sync)

    def count_sync(self) -> int:
        if self._collection is None:
            return 0
        with self._lock:
            return self._collection.count()

    def check_document_exists(self, source_file: str) -> bool:
        """Check if a source_file has already been ingested."""
        return self.count_document(source_file) > 0

    def count_document(self, source_file: str) -> int:
        """Return the number of indexed chunks belonging to a document."""
        if self._collection is None or self.count_sync() == 0:
            return 0
        try:
            with self._lock:
                results = self._collection.get(where={"source_file": source_file})
            return len(results["ids"])
        except Exception:
            traceback.print_exc()
            raise

    def list_collections(self) -> list[str]:
        if self._client is None:
            return []
        return [c.name for c in self._client.list_collections()]

    def close(self) -> None:
        self._collection = None
        self._client = None
