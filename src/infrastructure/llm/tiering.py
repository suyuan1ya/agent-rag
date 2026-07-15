"""Model Tier Router — dispatches LLM calls to the right model based on task type.

Three tiers with different cost/quality trade-offs:
  - Budget: cheap model for simple transformations (query rewrite, decomposition)
  - Standard: mid-tier model for evaluation tasks (sufficiency, citation verification)
  - Premium: best model for final answer generation and legal analysis

Token costs are tracked for each tier and aggregated into per-query cost reporting.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ModelTier(enum.Enum):
    BUDGET = "budget"       # cheapest, for simple rewrites
    STANDARD = "standard"   # balanced, for evaluation
    PREMIUM = "premium"     # best quality, for final answers


@dataclass
class TierConfig:
    """Configuration for one model tier."""
    model: str
    input_price_per_1k: float   # CNY per 1K input tokens
    output_price_per_1k: float  # CNY per 1K output tokens
    max_tokens_per_call: int
    description: str


# Default tier configurations (prices as of 2025H2, approximate)
DEFAULT_TIERS: dict[ModelTier, TierConfig] = {
    ModelTier.BUDGET: TierConfig(
        model="deepseek-chat",
        input_price_per_1k=0.001,    # ~1 RMB per 1M tokens
        output_price_per_1k=0.002,
        max_tokens_per_call=4096,
        description="DeepSeek Chat — cheapest, good for query rewriting",
    ),
    ModelTier.STANDARD: TierConfig(
        model="qwen-plus",
        input_price_per_1k=0.002,
        output_price_per_1k=0.006,
        max_tokens_per_call=8192,
        description="Qwen Plus — balanced, good for evaluation",
    ),
    ModelTier.PREMIUM: TierConfig(
        model="qwen-max",
        input_price_per_1k=0.02,
        output_price_per_1k=0.06,
        max_tokens_per_call=16384,
        description="Qwen Max — best quality, for final answers and legal analysis",
    ),
}

# Task-to-tier mapping
TASK_TIER_MAP: dict[str, ModelTier] = {
    # Query transformation tasks → BUDGET
    "rewrite_query": ModelTier.BUDGET,
    "decompose_question": ModelTier.BUDGET,
    "generate_hypothetical_answer": ModelTier.BUDGET,
    # Evaluation tasks → STANDARD
    "evaluate_sufficiency": ModelTier.STANDARD,
    "verify_citation": ModelTier.STANDARD,
    "intent_analysis": ModelTier.STANDARD,
    # Generation tasks → PREMIUM
    "answer_generation": ModelTier.PREMIUM,
    "clause_extraction": ModelTier.PREMIUM,
    "risk_assessment": ModelTier.PREMIUM,
    "comparison": ModelTier.PREMIUM,
}


class ModelTierRouter:
    """Routes LLM tasks to appropriate model tiers based on task type.

    Usage:
        router = ModelTierRouter()
        tier_config = router.route("rewrite_query")
        # → TierConfig(model="deepseek-chat", ...)

        # Record actual usage for cost tracking
        router.record_usage("rewrite_query", input_tokens=500, output_tokens=200)
        print(router.cost_summary)  # → {"total_cost_cny": 0.0009, ...}
    """

    def __init__(self, tiers: dict[ModelTier, TierConfig] | None = None):
        self._tiers = tiers or DEFAULT_TIERS
        self._task_map = dict(TASK_TIER_MAP)
        self._usage: list[_UsageRecord] = []
        self._total_cost: float = 0.0

    def route(self, task: str) -> TierConfig:
        """Get the tier config for a given task name.

        Falls back to STANDARD tier for unknown tasks.
        """
        tier = self._task_map.get(task, ModelTier.STANDARD)
        return self._tiers[tier]

    def get_model_for_task(self, task: str) -> str:
        """Get the model name to use for a given task."""
        return self.route(task).model

    def get_tier_for_task(self, task: str) -> ModelTier:
        """Get the tier enum for a given task."""
        return self._task_map.get(task, ModelTier.STANDARD)

    def set_task_tier(self, task: str, tier: ModelTier) -> None:
        """Override the default tier for a task."""
        self._task_map[task] = tier

    def record_usage(
        self,
        task: str,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
    ) -> None:
        """Record token consumption for cost tracking.

        Args:
            task: task name ("rewrite_query", "answer_generation", etc.)
            input_tokens: number of prompt tokens consumed
            output_tokens: number of completion tokens consumed
            model: actual model used (defaults to tier's model)
        """
        tier_config = self.route(task)
        if model is None:
            model = tier_config.model

        cost = (
            input_tokens / 1000 * tier_config.input_price_per_1k
            + output_tokens / 1000 * tier_config.output_price_per_1k
        )
        self._total_cost += cost
        self._usage.append(_UsageRecord(
            task=task,
            tier=self.get_tier_for_task(task).value,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_cny=cost,
        ))

    @property
    def total_cost_cny(self) -> float:
        return self._total_cost

    @property
    def cost_summary(self) -> dict:
        """Return cost breakdown by tier and task."""
        by_tier: dict[str, float] = {}
        by_task: dict[str, float] = {}
        total_tokens = 0

        for u in self._usage:
            by_tier[u.tier] = by_tier.get(u.tier, 0.0) + u.cost_cny
            by_task[u.task] = by_task.get(u.task, 0.0) + u.cost_cny
            total_tokens += u.input_tokens + u.output_tokens

        return {
            "total_cost_cny": round(self._total_cost, 6),
            "total_tokens": total_tokens,
            "num_calls": len(self._usage),
            "cost_by_tier": {k: round(v, 6) for k, v in by_tier.items()},
            "cost_by_task": {k: round(v, 6) for k, v in by_task.items()},
        }

    def reset(self) -> None:
        """Reset usage tracking (e.g., for a new query)."""
        self._usage.clear()
        self._total_cost = 0.0


@dataclass
class _UsageRecord:
    task: str
    tier: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_cny: float
