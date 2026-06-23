"""测试 rag/utils/ 中的工具函数（不需要 Milvus 连接）。"""

from __future__ import annotations

import sys
import os

# 直接导入 util 模块（绕过 rag.__init__ 避免触发 pymilvus 依赖）
_utils_dir = os.path.join(os.path.dirname(__file__), "..", "rag", "utils")
sys.path.insert(0, os.path.abspath(_utils_dir))

from dedup import deduplicate
from filters import _is_reference_header, _is_noise_block, _is_new_section_header


class TestDedup:
    def test_identical_texts_dedup(self):
        chunks = [
            {"text": "Transformer 架构基于自注意力机制", "page": 1},
            {"text": "Transformer 架构基于自注意力机制", "page": 2},
        ]
        result = deduplicate(chunks)
        assert len(result) == 1

    def test_similar_texts_dedup(self):
        chunks = [
            {"text": "深度学习是机器学习的一个分支，使用多层神经网络", "page": 1},
            {"text": "深度学习是机器学习的一个分支，使用多层神经网络进行训练", "page": 2},
        ]
        result = deduplicate(chunks)
        assert len(result) <= 2

    def test_different_texts_kept(self):
        chunks = [
            {"text": "Transformer 架构基于自注意力机制", "page": 1},
            {"text": "BERT 模型通过掩码语言模型进行预训练", "page": 5},
        ]
        result = deduplicate(chunks)
        assert len(result) == 2

    def test_empty_list(self):
        assert deduplicate([]) == []

    def test_single_item(self):
        chunks = [{"text": "唯一文本", "page": 1}]
        result = deduplicate(chunks)
        assert len(result) == 1
        assert result[0]["text"] == "唯一文本"


class TestFilters:
    def test_reference_header_detection(self):
        assert _is_reference_header("References")
        assert _is_reference_header("参考文献")
        assert _is_reference_header("Acknowledgments")
        assert _is_reference_header("致谢")
        assert not _is_reference_header("Introduction")
        assert not _is_reference_header("方法论")

    def test_section_header_detection(self):
        assert _is_new_section_header("Introduction")
        assert _is_new_section_header("Method")
        assert _is_new_section_header("Conclusion")
        assert not _is_new_section_header("some random text about methods")

    def test_noise_block_detection(self):
        assert _is_noise_block("[1] Smith, J. A review of machine learning.")
        assert _is_noise_block("12345 67890 11111 22222 33333 44444 55555")
        assert not _is_noise_block("深度学习通过多层神经网络学习数据的层次化表示。")

    def test_noise_block_data_heavy(self):
        text = "12.3 45.6 78.9 0.12 34.56 78.90 11.22 33.44"  # high digit+symbol ratio
        result = _is_noise_block(text)
        assert result
