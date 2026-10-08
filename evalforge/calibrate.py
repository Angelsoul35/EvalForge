"""Calibrate a judge against human labels.

A judge you have not measured is a judge you cannot trust. Given human scores
for the same cases and metrics, this computes, per metric, the Pearson and
Spearman correlation between judge and human scores and the mean absolute
error. Low correlation means the judge disagrees with people and its automated
verdicts should not gate anything important.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from evalforge.schemas import SuiteReport


@dataclass
class MetricCalibration:
    metric: str
    n: int
    pearson: float
    spearman: float
    mae: float


def calibrate(
    report: SuiteReport, human_labels: dict[str, dict[str, float]]
) -> list[MetricCalibration]:
    """Compare judge scores in ``report`` with ``human_labels[case_id][metric]``."""
    rows: list[dict] = []
    for result in report.results:
        human_for_case = human_labels.get(result.case_id, {})
        for metric, score in result.scores.items():
            if metric in human_for_case:
                rows.append(
                    {
                        "metric": metric,
                        "judge": score.score,
                        "human": float(human_for_case[metric]),
                    }
                )
    if not rows:
        return []

    frame = pd.DataFrame(rows)
    out: list[MetricCalibration] = []
    for metric, group in frame.groupby("metric"):
        judge = group["judge"]
        human = group["human"]
        out.append(
            MetricCalibration(
                metric=str(metric),
                n=len(group),
                pearson=_safe_corr(judge, human, "pearson"),
                spearman=_safe_corr(judge, human, "spearman"),
                mae=float((judge - human).abs().mean()),
            )
        )
    return sorted(out, key=lambda c: c.metric)


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    """Correlation that degrades gracefully when a series has zero variance."""
    if len(a) < 2 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.corr(b, method=method))
