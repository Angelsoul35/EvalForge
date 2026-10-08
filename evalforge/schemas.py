"""Data models for evaluation suites and results.

A *suite* is a list of *cases*. Each case has one question, the contexts the
retriever returned (each one marked relevant or not), the answer the system
gave, and an optional ground-truth answer. Running a suite gives a
``SuiteReport`` with the score of each metric per case and the average of each
metric over all cases.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, model_validator


class Context(BaseModel):
    """A single retrieved context passage and whether it is actually relevant."""

    text: str
    relevant: bool = False


class EvalCase(BaseModel):
    """One evaluation case."""

    id: str
    question: str
    answer: str
    contexts: list[Context] = Field(default_factory=list)
    ground_truth: str | None = None
    # Total number of relevant passages in the corpus for this question. When
    # provided, recall is computed against it; otherwise recall is reported over
    # the relevant passages that were retrieved.
    total_relevant: int | None = None

    @property
    def context_texts(self) -> list[str]:
        return [c.text for c in self.contexts]

    @property
    def relevant_texts(self) -> list[str]:
        return [c.text for c in self.contexts if c.relevant]


class Suite(BaseModel):
    name: str
    description: str = ""
    cases: list[EvalCase]

    @model_validator(mode="after")
    def _unique_ids(self) -> Suite:
        ids = [c.id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique within a suite")
        return self


class MetricScore(BaseModel):
    name: str
    score: float
    reason: str = ""


class CaseResult(BaseModel):
    case_id: str
    scores: dict[str, MetricScore]

    def value(self, metric: str) -> float:
        return self.scores[metric].score


class SuiteReport(BaseModel):
    suite_name: str
    judge: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    n_cases: int
    results: list[CaseResult]
    aggregates: dict[str, float]

    def aggregate(self, metric: str) -> float:
        return self.aggregates[metric]
