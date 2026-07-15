"""Prometheus metrics — agent, retrieval, LLM, and business-level instrumentation.

All metrics gracefully degrade to no-ops when prometheus_client is not installed.
"""

from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram

    # ── Agent metrics ──────────────────────────────────────

    agent_turns = Counter(
        "agent_turns_total",
        "Total agent FSM turns executed",
        ["state", "outcome"],  # state: intent_analysis|retrieval|evaluation|..., outcome: success|error
    )

    agent_queries = Counter(
        "agent_queries_total",
        "Total queries processed",
        ["intent", "status"],  # intent: qa|clause_extraction|..., status: completed|degraded|error
    )

    agent_fsm_transitions = Counter(
        "agent_fsm_transitions_total",
        "FSM state transitions",
        ["from_state", "to_state"],
    )

    # ── Tool metrics ───────────────────────────────────────

    tool_calls = Counter(
        "tool_calls_total",
        "Tool invocation count",
        ["tool_name", "status"],  # status: success|error
    )

    tool_latency = Histogram(
        "tool_latency_seconds",
        "Tool execution latency",
        ["tool_name"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )

    # ── Retrieval metrics ──────────────────────────────────

    retrieval_latency = Histogram(
        "retrieval_latency_seconds",
        "Retrieval operation latency",
        ["strategy"],  # dense|keyword|hybrid
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    retrieval_results = Histogram(
        "retrieval_results_count",
        "Number of results returned per retrieval",
        ["strategy"],
        buckets=(0, 1, 2, 3, 5, 10, 20, 50),
    )

    # ── LLM metrics ────────────────────────────────────────

    llm_tokens = Counter(
        "llm_tokens_total",
        "LLM token usage",
        ["model", "tier", "type"],  # type: prompt|completion
    )

    llm_latency = Histogram(
        "llm_latency_seconds",
        "LLM API call latency",
        ["model", "tier"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )

    llm_cost = Counter(
        "llm_cost_cny_total",
        "LLM API cost in CNY",
        ["model", "tier", "task"],
    )

    # ── Business metrics ───────────────────────────────────

    cost_per_query = Histogram(
        "cost_per_query_cny",
        "Total cost per query in CNY",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    )

    cache_hit_rate = Gauge(
        "cache_hit_rate",
        "Cache hit rate (0.0 - 1.0)",
        ["cache_type"],  # embedding|result
    )

    degradation_events = Counter(
        "degradation_events_total",
        "Degradation path activations",
        ["context"],  # reranker|chromadb|embedding|llm_api
    )

    circuit_breaker_state = Gauge(
        "circuit_breaker_state",
        "Circuit breaker state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)",
        ["name"],
    )

    token_budget_remaining = Gauge(
        "token_budget_remaining",
        "Remaining token budget for current query",
    )

    active_conversations = Gauge(
        "active_conversations",
        "Number of active conversations",
    )

    # ── System metrics ─────────────────────────────────────

    document_count = Gauge(
        "document_count_total",
        "Total indexed documents",
    )

    chunk_count = Gauge(
        "chunk_count_total",
        "Total indexed chunks",
    )

    # ── Helpers ────────────────────────────────────────────

    def record_agent_turn(state: str, outcome: str = "success"):
        agent_turns.labels(state=state, outcome=outcome).inc()

    def record_agent_query(intent: str, status: str):
        agent_queries.labels(intent=intent, status=status).inc()

    def record_fsm_transition(from_state: str, to_state: str):
        agent_fsm_transitions.labels(from_state=from_state, to_state=to_state).inc()

    def record_tool_call(tool_name: str, success: bool):
        tool_calls.labels(tool_name=tool_name, status="success" if success else "error").inc()

    def record_degradation(context: str):
        degradation_events.labels(context=context).inc()

    def update_circuit_breaker(name: str, state: str):
        state_map = {"closed": 0, "open": 1, "half_open": 2}
        circuit_breaker_state.labels(name=name).set(state_map.get(state, -1))

    def record_llm_usage(model: str, tier: str, prompt_tokens: int, completion_tokens: int, cost_cny: float = 0.0):
        llm_tokens.labels(model=model, tier=tier, type="prompt").inc(prompt_tokens)
        llm_tokens.labels(model=model, tier=tier, type="completion").inc(completion_tokens)
        if cost_cny > 0:
            llm_cost.labels(model=model, tier=tier, task="").inc(cost_cny)

    def update_cache_stats(embedding_hit_rate: float, result_hit_rate: float):
        cache_hit_rate.labels(cache_type="embedding").set(embedding_hit_rate)
        cache_hit_rate.labels(cache_type="result").set(result_hit_rate)

except ImportError:
    # Graceful degradation — all calls become no-ops
    def record_agent_turn(*args, **kwargs): pass
    def record_agent_query(*args, **kwargs): pass
    def record_fsm_transition(*args, **kwargs): pass
    def record_tool_call(*args, **kwargs): pass
    def record_degradation(*args, **kwargs): pass
    def update_circuit_breaker(*args, **kwargs): pass
    def record_llm_usage(*args, **kwargs): pass
    def update_cache_stats(*args, **kwargs): pass
    agent_turns = None
    agent_queries = None
    agent_fsm_transitions = None
    tool_calls = None
    tool_latency = None
    retrieval_latency = None
    retrieval_results = None
    llm_tokens = None
    llm_latency = None
    llm_cost = None
    cost_per_query = None
    cache_hit_rate = None
    degradation_events = None
    circuit_breaker_state = None
    token_budget_remaining = None
    active_conversations = None
    document_count = None
    chunk_count = None
