"""Async Circuit Breaker — prevents cascading failures from degraded dependencies.

Three-state design: CLOSED → OPEN → HALF_OPEN → CLOSED (or back to OPEN).

CLOSED:   Normal operation. Failures increment a counter.
OPEN:     After failure_threshold consecutive failures, the circuit opens.
          All calls fail fast with CircuitOpenError for recovery_timeout seconds.
HALF_OPEN: After recovery_timeout, a limited number of trial calls are allowed.
           Success → CLOSED. Failure → OPEN.
"""

from __future__ import annotations

import enum
import time
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is OPEN."""
    def __init__(self, name: str):
        super().__init__(f"Circuit '{name}' is OPEN — calls are blocked")
        self.name = name


class CircuitBreaker:
    """Async circuit breaker with sliding-window failure counting.

    Usage:
        cb = CircuitBreaker("reranker", failure_threshold=3, recovery_timeout=30.0)

        try:
            result = await cb.call(some_async_function, arg1, arg2)
        except CircuitOpenError:
            # Circuit is open, use fallback
            result = fallback_function(arg1, arg2)

    Attributes:
        name: unique name for this breaker (used in metrics/logging)
        failure_threshold: consecutive failures before opening
        recovery_timeout: seconds to wait before trying HALF_OPEN
        half_open_max_calls: max trial calls allowed in HALF_OPEN state
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 2,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")

        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0
        self._total_failures = 0
        self._total_successes = 0

    # ── public API ──────────────────────────────────────────

    async def call(
        self,
        fn: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute fn() with circuit breaker protection.

        Args:
            fn: async callable to protect
            *args, **kwargs: forwarded to fn

        Returns:
            fn's return value on success

        Raises:
            CircuitOpenError: if circuit is OPEN
            Original exception: if fn fails (re-raised after counting failure)
        """
        # Check / transition state before calling
        self._pre_call_check()

        try:
            result = await fn(*args, **kwargs)
            self._on_success()
            return result
        except CircuitOpenError:
            raise
        except Exception:
            self._on_failure()
            raise

    def call_sync(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Synchronous variant for non-async callables."""
        self._pre_call_check()

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except CircuitOpenError:
            raise
        except Exception:
            self._on_failure()
            raise

    # ── state management ────────────────────────────────────

    def _pre_call_check(self) -> None:
        """Check circuit state before allowing a call through."""
        if self.state == CircuitState.CLOSED:
            return

        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
            else:
                raise CircuitOpenError(self.name)

        # HALF_OPEN: allow up to half_open_max_calls trial calls
        if self.state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                raise CircuitOpenError(self.name)
            self._half_open_calls += 1

    def _on_success(self) -> None:
        """Record a successful call and potentially close the circuit."""
        self._total_successes += 1

        if self.state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        elif self.state == CircuitState.CLOSED:
            self._failure_count = 0  # reset on success

    def _on_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self._total_failures += 1
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self.state == CircuitState.HALF_OPEN:
            # Single failure in HALF_OPEN → back to OPEN
            self.state = CircuitState.OPEN
            self._success_count = 0
        elif self._failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    # ── introspection ───────────────────────────────────────

    def reset(self) -> None:
        """Force-reset the circuit to CLOSED (for testing/admin)."""
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
        }
