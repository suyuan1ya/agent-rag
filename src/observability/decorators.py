"""Instrumentation decorators — wire Prometheus metrics and OTel tracing into components.

These decorators are the glue between the infrastructure code and observability.
Apply them to tool functions, retrieval methods, and LLM calls to get automatic
metrics and tracing without modifying business logic.

All decorators gracefully degrade to no-ops when their respective libraries
are not installed.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Awaitable, Callable

from .metrics import (
    llm_latency,
    record_agent_turn,
    record_degradation,
    record_llm_usage,
    record_tool_call,
    retrieval_latency,
    retrieval_results,
    token_budget_remaining,
    tool_latency,
    update_circuit_breaker,
)
from .tracing import traced_span

# ── Tool instrumentation ───────────────────────────────────

def instrument_tool(tool_name: str):
    """Decorator for async tool functions: records count, latency, success/failure.

    Usage:
        @instrument_tool("hybrid_search")
        async def hybrid_search(query: str, top_k: int = 5) -> ToolResult:
            ...
    """
    def decorator(fn: Callable[..., Awaitable[Any]]):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            with traced_span(f"tool.{tool_name}", tool=tool_name, **{
                k: v for k, v in kwargs.items() if isinstance(v, (str, int, float))
            }):
                try:
                    result = await fn(*args, **kwargs)
                    duration = time.perf_counter() - start
                    success = getattr(result, "success", True)
                    record_tool_call(tool_name, success=success)
                    if tool_latency is not None:
                        tool_latency.labels(tool_name=tool_name).observe(duration)
                    return result
                except Exception:
                    duration = time.perf_counter() - start
                    record_tool_call(tool_name, success=False)
                    if tool_latency is not None:
                        tool_latency.labels(tool_name=tool_name).observe(duration)
                    raise
        return wrapper
    return decorator


# ── Retrieval instrumentation ──────────────────────────────

def instrument_retrieval(strategy: str):
    """Decorator for retrieval functions: records latency and result count.

    Usage:
        @instrument_retrieval("hybrid")
        async def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
            ...
    """
    def decorator(fn: Callable[..., Awaitable[Any]]):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            with traced_span(f"retrieval.{strategy}", strategy=strategy):
                try:
                    result = await fn(*args, **kwargs)
                    duration = time.perf_counter() - start
                    if retrieval_latency is not None:
                        retrieval_latency.labels(strategy=strategy).observe(duration)
                    count = len(result) if isinstance(result, list) else 0
                    if retrieval_results is not None:
                        retrieval_results.labels(strategy=strategy).observe(count)
                    return result
                except Exception:
                    duration = time.perf_counter() - start
                    if retrieval_latency is not None:
                        retrieval_latency.labels(strategy=strategy).observe(duration)
                    raise
        return wrapper
    return decorator


# ── LLM call instrumentation ───────────────────────────────

def instrument_llm_call(model: str = "unknown", tier: str = "standard"):
    """Decorator for LLM API calls: records latency, token usage, cost.

    Usage:
        @instrument_llm_call(model="qwen-plus", tier="standard")
        async def chat_completion(messages, **kwargs) -> dict:
            ...
    """
    def decorator(fn: Callable[..., Awaitable[Any]]):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            with traced_span("llm.call", model=model, tier=tier):
                try:
                    result = await fn(*args, **kwargs)
                    duration = time.perf_counter() - start
                    if llm_latency is not None:
                        llm_latency.labels(model=model, tier=tier).observe(duration)

                    usage = result.get("usage", {}) if isinstance(result, dict) else {}
                    if usage:
                        record_llm_usage(
                            model=model,
                            tier=tier,
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                        )
                    return result
                except Exception:
                    duration = time.perf_counter() - start
                    if llm_latency is not None:
                        llm_latency.labels(model=model, tier=tier).observe(duration)
                    raise
        return wrapper
    return decorator


# ── Agent FSM instrumentation ──────────────────────────────

def instrument_fsm_state(state_name: str):
    """Context manager decorator for FSM state execution.

    Usage:
        @instrument_fsm_state("retrieval")
        async def execute_retrieval(ctx):
            ...
    """
    def decorator(fn: Callable[..., Awaitable[Any]]):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            with traced_span(f"agent.{state_name}", state=state_name):
                try:
                    result = await fn(*args, **kwargs)
                    record_agent_turn(state=state_name, outcome="success")
                    return result
                except Exception:
                    record_agent_turn(state=state_name, outcome="error")
                    raise
        return wrapper
    return decorator


# ── Degradation event logger ───────────────────────────────

def log_degradation(context: str):
    """Record a degradation event in metrics."""
    record_degradation(context)


def update_cb_state(name: str, state: str):
    """Update circuit breaker state gauge."""
    update_circuit_breaker(name, state)


def update_budget_gauge(remaining: int):
    """Update token budget gauge."""
    if token_budget_remaining is not None:
        token_budget_remaining.set(remaining)
