"""CI quality gate: fail a build when eval metrics fall below thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field

from evalforge.schemas import SuiteReport


@dataclass
class GateFailure:
    metric: str
    actual: float | None
    threshold: float

    @property
    def reason(self) -> str:
        if self.actual is None:
            return f"{self.metric}: not measured in this suite (threshold {self.threshold})"
        return f"{self.metric}: {self.actual:.3f} < {self.threshold:.3f}"


@dataclass
class GateResult:
    passed: bool
    failures: list[GateFailure] = field(default_factory=list)


def evaluate_gate(report: SuiteReport, thresholds: dict[str, float]) -> GateResult:
    """Pass only if every thresholded metric meets its minimum aggregate score."""
    failures: list[GateFailure] = []
    for metric, minimum in thresholds.items():
        actual = report.aggregates.get(metric)
        if actual is None or actual < minimum:
            failures.append(GateFailure(metric=metric, actual=actual, threshold=minimum))
    return GateResult(passed=not failures, failures=failures)
