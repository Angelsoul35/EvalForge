"""The judge interface and the set of generation metrics judges can score."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evalforge.schemas import EvalCase, MetricScore

# Metrics a judge is responsible for. Retrieval metrics are label-based and live
# in evalforge.metrics.retrieval, not here.
GENERATION_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevance",
    "answer_correctness",
)

# answer_correctness needs a ground-truth answer; the others do not.
REQUIRES_GROUND_TRUTH: frozenset[str] = frozenset({"answer_correctness"})


@runtime_checkable
class Judge(Protocol):
    name: str

    def score(self, case: EvalCase, metric: str) -> MetricScore:
        """Score ``case`` on ``metric`` and return a value in [0, 1] with a reason."""
        ...
