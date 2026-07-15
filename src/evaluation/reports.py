"""Evaluation report generation — JSON and HTML output with statistical details.

Produces self-contained reports that document:
  - Dataset size, sampling method, metric definitions
  - Per-strategy metrics with 95% bootstrap CIs
  - Paired comparison results with p-values
  - Statistical methodology description
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .statistical import (
    bootstrap_confidence_interval,
    effect_size_cohens_d,
    paired_bootstrap_test,
)


@dataclass
class StrategyReport:
    """Metrics for a single retrieval strategy."""
    name: str
    recall_at_5: float
    recall_at_5_ci: tuple[float, float]  # (lower, upper) 95% CI
    mrr: float
    ndcg_at_5: float
    n_queries: int


@dataclass
class ComparisonReport:
    """Pairwise comparison of two strategies."""
    strategy_a: str
    strategy_b: str
    metric: str
    delta: float
    p_value: float
    significant: bool
    effect_size: float
    effect_size_interpretation: str


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    dataset: str
    dataset_size: int
    methodology_notes: list[str]
    strategies: list[StrategyReport]
    comparisons: list[ComparisonReport]
    ragas_metrics: dict[str, Any] = field(default_factory=dict)
    llm_judge_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "dataset": self.dataset,
            "dataset_size": self.dataset_size,
            "methodology": self.methodology_notes,
            "strategies": [
                {
                    "name": s.name,
                    "recall_at_5": round(s.recall_at_5, 4),
                    "recall_at_5_95ci": [round(s.recall_at_5_ci[0], 4), round(s.recall_at_5_ci[1], 4)],
                    "mrr": round(s.mrr, 4),
                    "ndcg_at_5": round(s.ndcg_at_5, 4),
                    "n_queries": s.n_queries,
                }
                for s in self.strategies
            ],
            "comparisons": [
                {
                    "strategies": f"{c.strategy_a} vs {c.strategy_b}",
                    "metric": c.metric,
                    "delta": round(c.delta, 4),
                    "p_value": round(c.p_value, 4),
                    "significant_at_005": c.significant,
                    "effect_size_cohens_d": round(c.effect_size, 3),
                    "interpretation": c.effect_size_interpretation,
                }
                for c in self.comparisons
            ],
            "ragas": self.ragas_metrics,
            "llm_judge": self.llm_judge_metrics,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def to_html(self) -> str:
        """Generate a self-contained HTML report."""
        data = self.to_dict()

        rows = ""
        for s in data["strategies"]:
            rows += f"""
            <tr>
                <td><strong>{s['name']}</strong></td>
                <td>{s['recall_at_5']:.4f} [{s['recall_at_5_95ci'][0]:.4f}, {s['recall_at_5_95ci'][1]:.4f}]</td>
                <td>{s['mrr']:.4f}</td>
                <td>{s['ndcg_at_5']:.4f}</td>
                <td>{s['n_queries']}</td>
            </tr>"""

        comp_rows = ""
        for c in data["comparisons"]:
            sig_badge = '<span style="color:green">SIGNIFICANT</span>' if c["significant_at_005"] else '<span style="color:gray">not significant</span>'
            comp_rows += f"""
            <tr>
                <td>{c['strategies']}</td>
                <td>{c['metric']}</td>
                <td>{c['delta']:+.4f}</td>
                <td>{c['p_value']:.4f}</td>
                <td>{sig_badge}</td>
                <td>{c['effect_size_cohens_d']:.3f} ({c['interpretation']})</td>
            </tr>"""

        method_items = "".join(f"<li>{m}</li>" for m in data["methodology"])

        return f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>RAG Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        h2 {{ border-bottom: 2px solid #333; padding-bottom: 4px; }}
        .meta {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>RAG Evaluation Report</h1>
    <p class="meta">Generated: {data['timestamp']} | Dataset: {data['dataset']} | Queries: {data['dataset_size']}</p>

    <h2>Methodology</h2>
    <ul>{method_items}</ul>

    <h2>Retrieval Metrics (with 95% Bootstrap CI)</h2>
    <table>
        <tr><th>Strategy</th><th>Recall@5 [95% CI]</th><th>MRR</th><th>NDCG@5</th><th>N</th></tr>
        {rows}
    </table>

    <h2>Pairwise Comparison (Paired Bootstrap Test)</h2>
    <table>
        <tr><th>Comparison</th><th>Metric</th><th>&Delta;</th><th>p-value</th><th>p&lt;0.05?</th><th>Cohen's d</th></tr>
        {comp_rows}
    </table>

    <h2>Metric Definitions</h2>
    <ul>
        <li><strong>Recall@k</strong>: Proportion of queries where at least one ground-truth relevant chunk appears in the top-k retrieved results.</li>
        <li><strong>MRR</strong>: Mean Reciprocal Rank — average of 1/rank of the first relevant result across all queries.</li>
        <li><strong>NDCG@k</strong>: Normalized Discounted Cumulative Gain — measures ranking quality with binary relevance labels.</li>
        <li><strong>95% CI</strong>: Bootstrap percentile confidence interval (Efron, 1994) with 1,000 resamples.</li>
        <li><strong>p-value</strong>: Two-sided paired bootstrap test (1,000 resamples). Significant at p &lt; 0.05.</li>
        <li><strong>Cohen's d</strong>: Effect size. d&lt;0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, &ge;0.8 large.</li>
    </ul>
</body>
</html>"""


# ── Report builder ─────────────────────────────────────────

class ReportBuilder:
    """Build an evaluation report from strategy scores."""

    def __init__(self, dataset_name: str, dataset_size: int):
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size
        self._strategy_scores: dict[str, dict[str, list[float]]] = {}
        self._methodology_notes: list[str] = [
            f"Dataset: {dataset_name}, {dataset_size} queries",
            "Retrieval metrics: Recall@k, MRR, NDCG@k with binary ground-truth relevance labels",
            "Confidence intervals: Bootstrap percentile method (Efron, 1994), 1,000 resamples, 95% CI",
            "Significance testing: Paired bootstrap test (Efron & Tibshirani, 1994), 1,000 resamples, two-sided",
            "Effect size: Cohen's d, interpreted per Cohen (1988) guidelines",
            "All random seeds fixed at 42 for reproducibility",
        ]

    def add_strategy_scores(
        self,
        name: str,
        recall5_scores: list[float],
        mrr_scores: list[float],
        ndcg5_scores: list[float],
    ) -> None:
        """Add per-query scores for a strategy."""
        self._strategy_scores[name] = {
            "recall@5": recall5_scores,
            "mrr": mrr_scores,
            "ndcg@5": ndcg5_scores,
        }

    def build(self) -> EvaluationReport:
        """Build the complete evaluation report."""
        # Build strategy reports
        strategy_reports = []
        for name, scores in self._strategy_scores.items():
            r5 = scores["recall@5"]
            mrr = scores["mrr"]
            n5 = scores["ndcg@5"]

            strategy_reports.append(StrategyReport(
                name=name,
                recall_at_5=sum(r5) / len(r5) if r5 else 0.0,
                recall_at_5_ci=bootstrap_confidence_interval(r5),
                mrr=sum(mrr) / len(mrr) if mrr else 0.0,
                ndcg_at_5=sum(n5) / len(n5) if n5 else 0.0,
                n_queries=len(r5),
            ))

        # Build pairwise comparisons (each strategy vs the first/baseline)
        baseline_name = list(self._strategy_scores.keys())[0] if self._strategy_scores else None
        comparisons = []
        if baseline_name:
            baseline_scores = self._strategy_scores[baseline_name]
            for name, scores in self._strategy_scores.items():
                if name == baseline_name:
                    continue
                for metric in ["recall@5", "mrr", "ndcg@5"]:
                    result = paired_bootstrap_test(
                        baseline_scores[metric], scores[metric]
                    )
                    d = effect_size_cohens_d(
                        baseline_scores[metric], scores[metric]
                    )
                    comparisons.append(ComparisonReport(
                        strategy_a=baseline_name,
                        strategy_b=name,
                        metric=metric,
                        delta=result.delta,
                        p_value=result.p_value,
                        significant=result.significant,
                        effect_size=d,
                        effect_size_interpretation=_interpret_d(d),
                    ))

        return EvaluationReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            dataset=self.dataset_name,
            dataset_size=self.dataset_size,
            methodology_notes=self._methodology_notes,
            strategies=strategy_reports,
            comparisons=comparisons,
        )


def _interpret_d(d: float) -> str:
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"
