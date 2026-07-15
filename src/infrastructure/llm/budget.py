"""Token Budget — per-query token limit with warning and exhaustion handling.

Prevents runaway LLM costs by enforcing a hard token cap per query.
When the budget is exhausted, the agent must fall back to cached responses
or return a partial answer with degradation notice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenConsumptionRecord:
    task: str
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: float


class TokenBudgetExhaustedError(Exception):
    """Raised when the token budget is exhausted."""
    def __init__(self, consumed: int, limit: int):
        super().__init__(f"Token budget exhausted: {consumed}/{limit}")
        self.consumed = consumed
        self.limit = limit


class TokenBudget:
    """Tracks token consumption against a per-query budget.

    Usage:
        budget = TokenBudget(max_tokens=50_000, warning_threshold=0.8)

        # Before an LLM call:
        if not budget.can_afford(estimated_tokens=2000):
            return degraded_response()

        # After the call:
        budget.consume(1500, model="qwen-plus", task="rewrite_query")

        if budget.exhausted:
            return early_termination()
    """

    def __init__(
        self,
        max_tokens: int = 50_000,
        warning_threshold: float = 0.8,
    ):
        if max_tokens < 100:
            raise ValueError("max_tokens must be >= 100")
        if not 0 < warning_threshold <= 1.0:
            raise ValueError("warning_threshold must be in (0, 1]")

        self.max_tokens = max_tokens
        self.warning_threshold = warning_threshold
        self._consumed: int = 0
        self._records: list[TokenConsumptionRecord] = []
        self._warned: bool = False

    @property
    def consumed(self) -> int:
        return self._consumed

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self._consumed)

    @property
    def exhausted(self) -> bool:
        return self._consumed >= self.max_tokens

    @property
    def usage_ratio(self) -> float:
        return self._consumed / self.max_tokens if self.max_tokens > 0 else 1.0

    def can_afford(self, estimated_tokens: int) -> bool:
        """Check if the estimated token count fits within remaining budget.

        Args:
            estimated_tokens: conservative estimate of input+output tokens
        """
        return self._consumed + estimated_tokens <= self.max_tokens

    def consume(
        self,
        tokens: int,
        model: str = "unknown",
        task: str = "unknown",
    ) -> None:
        """Record actual token consumption.

        Logs a warning when usage exceeds the warning threshold.
        Does NOT raise — callers should check `exhausted` and handle accordingly.
        """
        self._consumed += tokens
        self._records.append(TokenConsumptionRecord(
            task=task,
            model=model,
            input_tokens=tokens,  # simplified; caller can split input/output
            output_tokens=0,
            timestamp=0.0,
        ))

        if not self._warned and self.usage_ratio >= self.warning_threshold:
            self._warned = True
            logger.warning(
                "token_budget_warning",
                extra={
                    "consumed": self._consumed,
                    "limit": self.max_tokens,
                    "usage_ratio": f"{self.usage_ratio:.1%}",
                    "remaining": self.remaining,
                },
            )

    def consume_chat_response(self, response_usage: dict, model: str = "unknown", task: str = "unknown") -> None:
        """Record consumption from an OpenAI chat completion response."""
        input_tokens = response_usage.get("prompt_tokens", 0)
        output_tokens = response_usage.get("completion_tokens", 0)
        total = response_usage.get("total_tokens", input_tokens + output_tokens)
        self.consume(total, model=model, task=task)

    def estimate_tokens(self, text: str) -> int:
        """Quick token count estimate (4 chars ≈ 1 token for Chinese, 4 chars ≈ 1 token for English)."""
        return max(1, len(text) // 2)

    def reset(self) -> None:
        """Reset for a new query."""
        self._consumed = 0
        self._records.clear()
        self._warned = False

    @property
    def summary(self) -> dict:
        return {
            "consumed": self._consumed,
            "limit": self.max_tokens,
            "remaining": self.remaining,
            "exhausted": self.exhausted,
            "usage_ratio": f"{self.usage_ratio:.1%}",
            "num_calls": len(self._records),
        }
