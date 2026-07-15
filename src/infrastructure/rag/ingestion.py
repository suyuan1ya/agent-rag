"""Document ingestion pipeline — PDF parsing, cleaning, chunking, and indexing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import traceback
from typing import Any

# ── text cleaning ────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r"(\w+)-\s*\r?\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7e一-鿿　-〟＀-￯]", "", text)
    return text.strip()


# ── section / noise detection ────────────────────────────────


def _is_reference_header(text: str) -> bool:
    """Detect reference/acknowledgement section headers."""
    text_stripped = text.strip()
    if len(text_stripped) > 60:
        return False
    patterns = [
        r"^参考文献$",
        r"^References?$",
        r"^Bibliography$",
        r"^致谢$",
        r"^Acknowledgements?$",
        r"^附录$",
        r"^Appendix",
        r"^补充材料$",
        r"^Supplementary\s*Material",
    ]
    return any(re.match(p, text_stripped, re.IGNORECASE) for p in patterns)


def _is_new_section_header(text: str) -> bool:
    """Detect main content section headers that end the reference section."""
    text_stripped = text.strip()
    if len(text_stripped) > 60:
        return False
    patterns = [
        r"^引言$",
        r"^Introduction$",
        r"^方法$",
        r"^Methods?$",
        r"^结果$",
        r"^Results?$",
        r"^讨论$",
        r"^Discussion$",
        r"^结论$",
        r"^Conclusion",
    ]
    return any(re.match(p, text_stripped, re.IGNORECASE) for p in patterns)


def _is_noise_block(text: str) -> bool:
    """Filter out citation-only blocks and high-noise content."""
    text_stripped = text.strip()
    if len(text_stripped) < 20:
        return True
    # Pure citation like "[1] Author. Title. Journal, 2020."
    if re.match(r"^\[\d+\][\s,;].{5,}", text_stripped):
        digits_and_symbols = sum(1 for c in text_stripped if c.isdigit() or c in "[](){},.;:/-")
        if digits_and_symbols / len(text_stripped) > 0.55:
            return True
    return False


# ── chunking ─────────────────────────────────────────────────


def build_chunks(
    slices: list[dict],
    chunk_size: int = 500,
    overlap: int = 150,
    min_chunk_len: int = 50,
) -> list[dict]:
    """Page-grouped sliding window chunking with natural boundary detection.

    Priority for split points:
      1. Paragraph breaks (double newline)
      2. Chinese sentence end (。/！/？)
      3. English sentence end (.!? followed by space + uppercase)
      4. Chinese semicolon (；)
      5. Space (fallback)
    """
    pages: dict[int, list[str]] = {}
    for s in slices:
        pages.setdefault(s["page_number"], []).append(s["text"])

    chunks: list[dict] = []
    for page_num, texts in pages.items():
        full_text = " ".join(texts)
        start = 0
        max_iterations = max(len(full_text), 1) + 100
        iterations = 0

        while start < len(full_text):
            iterations += 1
            if iterations > max_iterations:
                break

            raw_end = min(start + chunk_size, len(full_text))
            if raw_end < len(full_text):
                window = min(200, chunk_size // 2)
                search_start = max(start, raw_end - window)
                search_end = min(len(full_text), raw_end + window)
                region = full_text[search_start:search_end]
                best = -1

                # Priority 1: paragraph break
                m = re.search(r"\n\s*\n", region)
                if m and search_start + m.start() < raw_end + 60:
                    best = search_start + m.start()

                # Priority 2: Chinese sentence end
                if best == -1:
                    for term in ["。", "！", "？"]:
                        pos = region.rfind(term, 0, raw_end - search_start + 40)
                        if pos != -1:
                            after = region[pos + 1 : pos + 4] if pos + 1 < len(region) else ""
                            if not after.startswith("）") and not after.startswith("」"):
                                best = search_start + pos + 1
                                break

                # Priority 3: English sentence end
                if best == -1:
                    m = re.search(r"[.!?]\s+[A-Z]", region)
                    if m:
                        best = search_start + m.start() + 1

                # Priority 4: Chinese semicolon
                if best == -1:
                    pos = region.rfind("；", 0, raw_end - search_start + 40)
                    if pos != -1:
                        best = search_start + pos + 1

                # Fallback: space
                if best == -1:
                    pos = region.rfind(
                        " ", raw_end - search_start - 60, raw_end - search_start + 60
                    )
                    if pos != -1:
                        best = search_start + pos

                if best != -1 and best > start:
                    raw_end = best

            chunk_text = full_text[start:raw_end].strip()
            if len(chunk_text) >= min_chunk_len:
                chunks.append({"text": chunk_text, "page_number": page_num})
                if raw_end >= len(full_text):
                    break
                start = max(raw_end - overlap, start + 1)
            else:
                start += chunk_size - overlap

    return chunks


# ── PDF extraction ──────────────────────────────────────────


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract and clean text blocks from a PDF using PyMuPDF.

    Returns:
        list[dict]: [{"text": str, "page_number": int}, ...]
    """
    import fitz

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception:
        traceback.print_exc()
        raise RuntimeError(f"Cannot open PDF: {pdf_path}")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", prefix="rag_raw_", text=True)
    element_count = 0

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp:
            for page_idx in range(len(pdf_doc)):
                page = pdf_doc[page_idx]
                text = clean_text(page.get_text("text"))
                if text:
                    page_num = page_idx + 1
                    tmp.write(json.dumps({"t": text, "p": page_num}, ensure_ascii=False) + "\n")
                    element_count += 1

                if (page_idx + 1) % 10 == 0:
                    print(f"  Parsed {page_idx + 1} pages, {element_count} elements...")
    finally:
        pdf_doc.close()

    # Stream-filter from temp file
    MIN_TEXT_LEN = 50
    filtered: list[dict] = []
    filtered_ref = filtered_noise = filtered_short = 0
    in_reference_section = False

    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                text = rec["t"]

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

                filtered.append({"text": text, "page_number": rec["p"]})
    finally:
        os.unlink(tmp_path)

    print(
        f"Cleaned: {element_count} raw → {len(filtered)} valid "
        f"(-{filtered_ref} refs, -{filtered_noise} noise, -{filtered_short} short)"
    )
    return filtered


# ── pipeline ─────────────────────────────────────────────────


class IngestionPipeline:
    """Orchestrates PDF ingestion: extract → clean → chunk → index.

    Coordinates between the vector store, embedding provider, and BM25 index
    to produce a fully searchable document set.
    """

    def __init__(
        self,
        vector_store: Any,  # VectorStore
        embedding_provider: Any,  # EmbeddingProvider
        bm25_index: Any,  # BM25Index
        chunk_size: int = 500,
        chunk_overlap: int = 150,
        batch_size: int = 100,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.bm25_index = bm25_index
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

    async def ingest(
        self,
        pdf_path: str,
        check_duplicates: bool = True,
        source_name: str | None = None,
    ) -> int:
        """Run the full ingestion pipeline for a PDF.

        Returns:
            int: number of chunks indexed
        """
        # 1. Check for existing ingestion
        source_name = source_name or os.path.abspath(pdf_path)
        if check_duplicates:
            try:
                existing_chunks = self.vector_store.count_document(source_name)
                if existing_chunks:
                    print(f"PDF already ingested: {pdf_path}")
                    return existing_chunks
            except Exception:
                print("Cannot verify ingestion status, proceeding...")

        # 2. Extract and clean text from PDF
        print(f"Extracting text from: {pdf_path}")
        slices = extract_text_from_pdf(pdf_path)

        if not slices:
            raise ValueError("No valid text extracted from PDF")

        # 3. Build chunks
        chunks = build_chunks(
            slices,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        print(f"Generated {len(chunks)} chunks")

        # 4. Build BM25 index
        self.bm25_index.build(self.bm25_index.get_chunks() + chunks)
        print(
            "BM25 index built: "
            f"{self.bm25_index.vocab_size} terms, "
            f"{self.bm25_index.doc_count} docs"
        )

        # 5. Encode and insert into vector store
        total_inserted = 0
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_texts = [c["text"] for c in batch]

            embeddings = await self.embedding_provider.encode(batch_texts, add_query_prefix=False)

            source_id = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:16]
            ids = [f"{source_id}_chunk_{i + j}" for j in range(len(batch))]
            metadatas = [
                {
                    "page_number": batch[j]["page_number"],
                    "source_file": source_name,
                }
                for j in range(len(batch))
            ]

            await self.vector_store.add(
                ids=ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=metadatas,
            )
            total_inserted += len(batch)

        print(f"Ingestion complete: {total_inserted} chunks indexed")
        return total_inserted
