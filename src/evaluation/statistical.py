"""Statistical significance testing for retrieval evaluation.

Methods:
  - Bootstrap confidence intervals (95% CI, 1000 resamples)
  - Paired bootstrap test for comparing two strategies (p < 0.05)

All methods are documented with formulas to ensure reproducibility.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable


@dataclass
class SignificanceResult:
    """Result of a statistical significance test."""
    method: str
    statistic_a: float       # Mean metric for strategy A
    statistic_b: float       # Mean metric for strategy B
    delta: float             # Difference (B - A)
    ci_lower: float          # 95% CI lower bound
    ci_upper: float          # 95% CI upper bound
    p_value: float            # Two-sided p-value
    significant: bool         # p < 0.05
    n_bootstrap: int = 1000
    n_samples: int = 0       # Number of queries in the test


def bootstrap_confidence_interval(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a mean metric.

    Method: Efron's percentile bootstrap (Efron & Tibshirani, 1994).
    1. Resample N scores with replacement, N = len(scores)
    2. Compute mean of each bootstrap sample
    3. Take the (alpha/2) and (1 - alpha/2) percentiles of bootstrap means

    Args:
        scores: list of per-query metric values
        n_bootstrap: number of bootstrap resamples (default 1000)
        ci_level: confidence level (default 0.95)
        seed: random seed for reproducibility

    Returns:
        (lower_bound, upper_bound)
    """
    if len(scores) < 5:
        return float("nan"), float("nan")

    random.seed(seed)
    n = len(scores)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = [scores[random.randint(0, n - 1)] for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    alpha = 1.0 - ci_level
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)

    return bootstrap_means[lower_idx], bootstrap_means[upper_idx]


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> SignificanceResult:
    """Paired bootstrap test for comparing two strategies on the same queries.

    Null hypothesis: strategy A and B have the same mean metric.

    Method (paired bootstrap):
    1. Compute observed delta = mean(B) - mean(A)
    2. Bootstrap N pairs: for each bootstrap sample, compute mean(B*) - mean(A*)
    3. Two-sided p-value = proportion of bootstrap deltas that are more extreme
       than the observed delta (i.e., |delta*| >= |observed_delta|)

    Args:
        scores_a: per-query metric for strategy A
        scores_b: per-query metric for strategy B (same queries, same order)
        n_bootstrap: number of bootstrap resamples
        seed: random seed

    Returns:
        SignificanceResult with p-value and confidence interval
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Paired test requires equal-length score lists. "
            f"Got {len(scores_a)} vs {len(scores_b)}"
        )
    if len(scores_a) < 5:
        raise ValueError(f"Need at least 5 samples, got {len(scores_a)}")

    random.seed(seed)
    n = len(scores_a)

    # Observed statistics
    mean_a = sum(scores_a) / n
    mean_b = sum(scores_b) / n
    observed_delta = mean_b - mean_a

    # Paired bootstrap
    bootstrap_deltas = []
    for _ in range(n_bootstrap):
        sum_a = 0.0
        sum_b = 0.0
        for _ in range(n):
            idx = random.randint(0, n - 1)
            sum_a += scores_a[idx]
            sum_b += scores_b[idx]
        bootstrap_deltas.append((sum_b - sum_a) / n)

    bootstrap_deltas.sort()

    # 95% CI for the delta
    lower_idx = int(0.025 * n_bootstrap)
    upper_idx = int(0.975 * n_bootstrap)
    ci_lower = bootstrap_deltas[lower_idx]
    ci_upper = bootstrap_deltas[upper_idx]

    # Two-sided p-value: proportion of bootstrap samples where |delta| >= |observed|
    more_extreme = sum(
        1 for d in bootstrap_deltas
        if abs(d - observed_delta) >= abs(observed_delta) or abs(d) >= abs(observed_delta)
    )
    p_value = more_extreme / n_bootstrap

    return SignificanceResult(
        method="paired_bootstrap",
        statistic_a=mean_a,
        statistic_b=mean_b,
        delta=observed_delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        significant=p_value < 0.05,
        n_bootstrap=n_bootstrap,
        n_samples=n,
    )


def effect_size_cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """Cohen's d effect size for paired samples.

    d = mean_difference / std_difference

    Interpretation:
      d < 0.2: negligible
      0.2 <= d < 0.5: small
      0.5 <= d < 0.8: medium
      d >= 0.8: large
    """
    n = len(scores_a)
    if n < 2:
        return float("nan")

    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / n

    if n == 1:
        return float("nan")

    variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_diff = math.sqrt(variance)

    if std_diff < 1e-10:
        return float("inf") if mean_diff != 0 else 0.0

    return mean_diff / std_diff


def compute_retrieval_metrics_with_ground_truth(
    test_cases: list,
    search_fn: Callable,
    top_k: int = 5,
) -> dict:
    """Compute Recall@k, MRR, NDCG@k using ground truth relevance labels.

    Unlike the Jaccard-similarity approach (metrics.py), this uses
    QueryTestCase.relevant_chunk_indices as binary ground truth labels.

    Metric definitions:
      Recall@k = |retrieved_relevant ∩ all_relevant| / |all_relevant|
      MRR = mean(1 / rank_of_first_relevant) across all queries
      NDCG@k = DCG@k / IDCG@k, where relevance is binary {0, 1}
    """
    recall_sums = {k: 0.0 for k in [1, 3, 5, 10] if k <= top_k}
    mrr_sum = 0.0
    ndcg_sums = {k: 0.0 for k in [1, 3, 5, 10] if k <= top_k}
    valid_queries = 0

    for case in test_cases:
        results = search_fn(case.question, top_k=top_k) or []

        # Binary relevance: check if any ground truth chunk is in results
        relevant_indices = set(case.relevant_chunk_indices)
        if not relevant_indices:
            continue

        valid_queries += 1
        retrieved_relevant = set()

        # Find rank of first relevant result
        first_rank = None
        for rank, r in enumerate(results):
            chunk_idx = r.get("chunk_index", -1)
            if chunk_idx in relevant_indices:
                retrieved_relevant.add(chunk_idx)
                if first_rank is None:
                    first_rank = rank + 1

        # Recall@k
        for k in recall_sums:
            k_results = set(
                r.get("chunk_index", -1) for r in results[:k]
                if r.get("chunk_index", -1) in relevant_indices
            )
            recall_sums[k] += len(k_results) / len(relevant_indices)

        # MRR
        if first_rank is not None:
            mrr_sum += 1.0 / first_rank

        # NDCG@k (binary relevance: 1 if relevant, 0 otherwise)
        for k in ndcg_sums:
            dcg = sum(
                1.0 / math.log2(rank + 2)
                for rank, r in enumerate(results[:k])
                if r.get("chunk_index", -1) in relevant_indices
            )
            # IDCG: ideal ranking has all relevant docs at the top
            ideal_count = min(len(relevant_indices), k)
            idcg = sum(
                1.0 / math.log2(i + 2)
                for i in range(ideal_count)
            )
            ndcg_sums[k] += dcg / idcg if idcg > 0 else 0.0

    n = max(valid_queries, 1)
    metrics = {}
    for k in recall_sums:
        metrics[f"Recall@{k}"] = recall_sums[k] / n
        metrics[f"NDCG@{k}"] = ndcg_sums[k] / n
    metrics["MRR"] = mrr_sum / n
    metrics["queries"] = valid_queries

    # Per-query scores for statistical testing
    metrics["_per_query_recall5"] = []  # populated by caller if needed

    return metrics
