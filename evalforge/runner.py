"""Run a suite: compute retrieval metrics and judge the generation metrics."""

from __future__ import annotations

from evalforge.judges.base import GENERATION_METRICS, REQUIRES_GROUND_TRUTH, Judge
from evalforge.metrics import retrieval
from evalforge.schemas import CaseResult, EvalCase, MetricScore, Suite, SuiteReport
from evalforge.tracing import span

RETRIEVAL_METRICS: tuple[str, ...] = (
    "hit_rate",
    "context_precision",
    "context_recall",
    "mrr",
)

_RETRIEVAL_FUNCS = {
    "hit_rate": retrieval.hit_rate,
    "context_precision": retrieval.context_precision,
    "context_recall": retrieval.context_recall,
    "mrr": retrieval.mrr,
}


class EvalRunner:
    def __init__(self, judge: Judge, otel_enabled: bool = False) -> None:
        self.judge = judge
        self.otel_enabled = otel_enabled

    def run_case(self, case: EvalCase) -> CaseResult:
        scores: dict[str, MetricScore] = {}

        if case.contexts:
            for name, func in _RETRIEVAL_FUNCS.items():
                scores[name] = MetricScore(
                    name=name, score=round(func(case), 4), reason="label-based"
                )

        for metric in GENERATION_METRICS:
            if metric in REQUIRES_GROUND_TRUTH and not case.ground_truth:
                continue
            verdict = self.judge.score(case, metric)
            scores[metric] = MetricScore(
                name=metric, score=round(verdict.score, 4), reason=verdict.reason
            )

        return CaseResult(case_id=case.id, scores=scores)

    def run(self, suite: Suite) -> SuiteReport:
        results: list[CaseResult] = []
        with span("evalforge.suite", self.otel_enabled, suite=suite.name, cases=len(suite.cases)):
            for case in suite.cases:
                with span("evalforge.case", self.otel_enabled, case_id=case.id):
                    results.append(self.run_case(case))

        return SuiteReport(
            suite_name=suite.name,
            judge=self.judge.name,
            n_cases=len(suite.cases),
            results=results,
            aggregates=_aggregate(results),
        )


def _aggregate(results: list[CaseResult]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for result in results:
        for name, metric in result.scores.items():
            sums[name] = sums.get(name, 0.0) + metric.score
            counts[name] = counts.get(name, 0) + 1
    return {name: round(sums[name] / counts[name], 4) for name in sorted(sums)}
