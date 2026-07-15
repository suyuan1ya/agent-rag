import pytest

from src.core.runtime import normalize_knowledge_base_id
from src.infrastructure.rag.bm25_index import _hamming
from src.infrastructure.rag.ingestion import build_chunks


def test_normalize_knowledge_base_id():
    assert normalize_knowledge_base_id("team-a") == "team-a"
    assert normalize_knowledge_base_id("team a") == "team-a"


@pytest.mark.parametrize("value", ["", "x" * 65])
def test_invalid_knowledge_base_id(value):
    with pytest.raises(ValueError):
        normalize_knowledge_base_id(value)


def test_hamming_is_compatible_with_python_39():
    assert _hamming(0b1010, 0b0011) == 2


def test_chunking_stops_after_the_final_window():
    chunks = build_chunks(
        [{"text": "甲" * 1000, "page_number": 1}],
        chunk_size=500,
        overlap=150,
    )

    assert len(chunks) == 3
    assert len({chunk["text"] for chunk in chunks}) <= 2
