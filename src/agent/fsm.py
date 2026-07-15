"""Agent Finite State Machine — explicit states and transition rules.

Replaces the simple for-loop in ResearchAgent with a proper FSM that provides:
  - Predictable state transitions (no runaway loops)
  - Serializable state for crash recovery
  - Observable state changes (logged + metered)
  - Explicit error state with degradation paths
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Callable


class AgentState(enum.Enum):
    """States in the agent reasoning lifecycle."""

    IDLE = "idle"  # Initial state, no work done
    INTENT_ANALYSIS = "intent_analysis"  # Classify user intent (qa / clause / risk / compare)
    STRATEGY_SELECTION = "strategy_selection"  # Choose tool sequence based on intent
    RETRIEVAL = "retrieval"  # Execute selected search/transform tools
    EVALUATION = "evaluation"  # Assess result sufficiency + citation accuracy
    REFINEMENT = "refinement"  # Rewrite/decompose and re-search (max 2 cycles)
    ANSWER_GENERATION = "answer_generation"  # Synthesize final answer from context
    DONE = "done"  # Terminal: success
    ERROR = "error"  # Terminal: unrecoverable failure


@dataclass
class Transition:
    """A possible transition from one state to another, guarded by a condition."""

    source: AgentState
    target: AgentState
    condition: Callable[["AgentContext"], bool] = field(default=lambda ctx: True)
    priority: int = 0  # Higher = evaluated first
    description: str = ""

    def can_fire(self, ctx: "AgentContext") -> bool:
        try:
            return self.condition(ctx)
        except Exception:
            return False


@dataclass
class AgentContext:
    """Mutable context carried through the FSM execution.

    Holds all state the agent accumulates during a query: messages, tool results,
    evaluation scores, token usage, error state, etc.
    """

    query: str = ""
    intent: str = "qa"  # qa | clause_extraction | risk_assessment | comparison
    messages: list[dict] = field(default_factory=list)
    tool_calls_made: int = 0
    total_tokens: int = 0
    strategy: list[str] = field(default_factory=list)  # ordered tool names to execute
    current_tool_index: int = 0
    retrieval_results: list[dict] = field(default_factory=list)
    evaluation_score: float = 0.0
    evaluation_sufficient: bool = False
    refinement_count: int = 0
    max_refinements: int = 2
    final_answer: str = ""
    sources: list[dict] = field(default_factory=list)
    error_message: str = ""
    degraded: bool = False
    degraded_reason: str = ""

    # Timestamps for latency tracking
    started_at: float = 0.0
    state_entered_at: float = 0.0

    # Cost tracking
    token_budget_consumed: int = 0
    token_budget_limit: int = 50_000


# ── Default transition table ─────────────────────────────────


def _always(ctx: AgentContext) -> bool:
    return True


def _intent_classified(ctx: AgentContext) -> bool:
    return bool(ctx.intent)


def _strategy_chosen(ctx: AgentContext) -> bool:
    return bool(ctx.strategy)


def _tools_exhausted(ctx: AgentContext) -> bool:
    return ctx.current_tool_index >= len(ctx.strategy)


def _tools_remaining(ctx: AgentContext) -> bool:
    return ctx.current_tool_index < len(ctx.strategy)


def _evaluation_sufficient(ctx: AgentContext) -> bool:
    return ctx.evaluation_sufficient


def _can_refine(ctx: AgentContext) -> bool:
    return ctx.refinement_count < ctx.max_refinements


def _has_error(ctx: AgentContext) -> bool:
    return bool(ctx.error_message)


DEFAULT_TRANSITIONS: list[Transition] = [
    # Normal flow
    Transition(AgentState.IDLE, AgentState.INTENT_ANALYSIS, _always, 10, "Start analysis"),
    Transition(
        AgentState.INTENT_ANALYSIS,
        AgentState.STRATEGY_SELECTION,
        _intent_classified,
        10,
        "Intent classified",
    ),
    Transition(
        AgentState.INTENT_ANALYSIS, AgentState.ERROR, _has_error, 100, "Intent analysis failed"
    ),
    Transition(
        AgentState.STRATEGY_SELECTION,
        AgentState.RETRIEVAL,
        _strategy_chosen,
        10,
        "Strategy selected",
    ),
    Transition(
        AgentState.STRATEGY_SELECTION,
        AgentState.ERROR,
        _has_error,
        100,
        "Strategy selection failed",
    ),
    Transition(
        AgentState.RETRIEVAL, AgentState.EVALUATION, _tools_exhausted, 10, "Tools exhausted"
    ),
    Transition(
        AgentState.RETRIEVAL,
        AgentState.RETRIEVAL,
        _tools_remaining,
        5,
        "Continue retrieval strategy",
    ),
    Transition(
        AgentState.RETRIEVAL, AgentState.ERROR, _has_error, 100, "All retrieval tools failed"
    ),
    Transition(
        AgentState.EVALUATION,
        AgentState.ANSWER_GENERATION,
        _evaluation_sufficient,
        10,
        "Sufficient results",
    ),
    Transition(AgentState.EVALUATION, AgentState.REFINEMENT, _can_refine, 8, "Need refinement"),
    Transition(
        AgentState.EVALUATION, AgentState.ANSWER_GENERATION, _always, 1, "Refinements exhausted"
    ),
    Transition(
        AgentState.REFINEMENT, AgentState.RETRIEVAL, _always, 10, "Re-search with refined query"
    ),
    Transition(AgentState.ANSWER_GENERATION, AgentState.DONE, _always, 10, "Answer generated"),
    Transition(
        AgentState.ANSWER_GENERATION, AgentState.ERROR, _has_error, 100, "Answer generation failed"
    ),
    # Terminal states have no outgoing transitions
]


# ── FSM engine ───────────────────────────────────────────────


class AgentStateMachine:
    """Event-driven FSM for the agent reasoning loop.

    Usage:
        fsm = AgentStateMachine()
        ctx = AgentContext(query="...")
        fsm.start(ctx)
        while not fsm.is_terminal():
            state = fsm.current_state
            # Execute state-specific logic externally...
            fsm.transition(ctx)  # evaluate guards and move to next state
    """

    def __init__(self, transitions: list[Transition] | None = None):
        self._transitions = transitions or DEFAULT_TRANSITIONS
        self._current_state = AgentState.IDLE
        self._state_history: list[tuple[AgentState, float]] = []
        self._started_at: float = 0.0

    @property
    def current_state(self) -> AgentState:
        return self._current_state

    @property
    def state_history(self) -> list[tuple[AgentState, float]]:
        """Ordered list of (state, timestamp) traversed."""
        return list(self._state_history)

    @property
    def elapsed_seconds(self) -> float:
        if self._started_at == 0:
            return 0.0
        return time.monotonic() - self._started_at

    def start(self, ctx: AgentContext) -> None:
        """Reset and start the FSM from IDLE."""
        self._current_state = AgentState.IDLE
        self._state_history = [(AgentState.IDLE, time.monotonic())]
        self._started_at = time.monotonic()
        ctx.started_at = self._started_at
        ctx.state_entered_at = self._started_at

    def transition(self, ctx: AgentContext) -> AgentState:
        """Evaluate guards and transition to the next state.

        Returns the new current state after transition.
        """
        # Find the highest-priority valid transition from current state
        candidates = [
            t for t in self._transitions if t.source == self._current_state and t.can_fire(ctx)
        ]
        candidates.sort(key=lambda t: t.priority, reverse=True)

        if not candidates:
            # No valid transition → stay in current state (or go to ERROR if stuck)
            if self._current_state not in (AgentState.DONE, AgentState.ERROR):
                ctx.error_message = f"No valid transition from {self._current_state.value}"
                self._current_state = AgentState.ERROR
        else:
            self._current_state = candidates[0].target

        now = time.monotonic()
        self._state_history.append((self._current_state, now))
        ctx.state_entered_at = now

        return self._current_state

    def force_state(self, state: AgentState, ctx: AgentContext) -> None:
        """Override current state (e.g., for error recovery)."""
        self._current_state = state
        now = time.monotonic()
        self._state_history.append((state, now))
        ctx.state_entered_at = now

    def is_terminal(self) -> bool:
        return self._current_state in (AgentState.DONE, AgentState.ERROR)

    def snapshot(self) -> dict:
        """Serialize FSM state for crash recovery."""
        return {
            "current_state": self._current_state.value,
            "history": [(s.value, ts) for s, ts in self._state_history],
            "elapsed": self.elapsed_seconds,
        }

    def restore(self, snapshot: dict) -> None:
        """Restore FSM state from a snapshot."""
        self._current_state = AgentState(snapshot["current_state"])
        self._state_history = [(AgentState(s), ts) for s, ts in snapshot.get("history", [])]
        self._started_at = time.monotonic() - snapshot.get("elapsed", 0)
