"""pytest 全局 fixtures。"""

from __future__ import annotations

import pytest


@pytest.fixture
def test_chunks() -> list[dict]:
    """合成测试用 chunk 列表。"""
    return [
        {
            "text": "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的层次化表示。",
            "page_number": 1,
        },
        {
            "text": "Transformer 架构基于自注意力机制，摒弃了传统的循环神经网络结构。",
            "page_number": 3,
        },
        {
            "text": "BERT 模型通过掩码语言模型（MLM）和下一句预测（NSP）进行预训练。",
            "page_number": 5,
        },
        {
            "text": "注意力机制的核心思想是让模型能够动态地关注输入序列中的不同部分。",
            "page_number": 7,
        },
        {
            "text": "大语言模型的涌现能力体现在少样本学习和思维链推理等方面。",
            "page_number": 10,
        },
    ]


@pytest.fixture
def mock_llm_responses():
    """模拟 LLM 响应。"""
    return {
        "plan": '{"sub_queries": [{"question": "什么是深度学习?", "strategy": "hybrid"}]}',
        "answer": "深度学习是机器学习的一个子领域，使用多层神经网络。",
        "decompose": '[{"question": "子问题1", "rationale": "理由"}]',
        "evaluate": '{"sufficient": true, "score": 0.8, "reasoning": "结果充足"}',
    }
