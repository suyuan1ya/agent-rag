"""Prometheus 指标定义。"""

from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram, Gauge

    agent_turns = Counter(
        "agent_turns_total",
        "Total agent inference turns",
        ["outcome"],  # success, error, max_iterations
    )

    tool_calls = Counter(
        "tool_calls_total",
        "Tool invocation count",
        ["tool_name", "status"],  # success, error
    )

    retrieval_latency = Histogram(
        "retrieval_latency_seconds",
        "Retrieval latency in seconds",
        ["strategy"],  # dense, keyword, hybrid
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    llm_tokens = Counter(
        "llm_tokens_total",
        "LLM token usage",
        ["model", "type"],  # prompt, completion
    )

    active_conversations = Gauge(
        "active_conversations",
        "Number of active conversations",
    )

    def record_tool_call(tool_name: str, success: bool):
        tool_calls.labels(tool_name=tool_name, status="success" if success else "error").inc()

    def record_agent_turn(outcome: str):
        agent_turns.labels(outcome=outcome).inc()

except ImportError:
    # prometheus_client 未安装时的空实现
    def record_tool_call(*args, **kwargs): pass
    def record_agent_turn(*args, **kwargs): pass
    agent_turns = None
    tool_calls = None
    retrieval_latency = None
    llm_tokens = None
    active_conversations = None
