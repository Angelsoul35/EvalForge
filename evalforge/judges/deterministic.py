"""A deterministic, offline judge.

It scores the generation metrics with lexical similarity (TF-IDF cosine) and
token F1. It is not as good as a strong LLM judge on nuance, but it is free,
fast, and gives the same result every time, so it is the right default for
tests, CI, and learning. It uses the same ``Judge`` interface as the Anthropic
judge, so switching between them is one line.
"""

from __future__ import annotations

from evalforge.judges.base import GENERATION_METRICS
from evalforge.metrics.similarity import (
    TfidfSimilarity,
    mean,
    split_sentences,
    token_f1,
)
from evalforge.schemas import EvalCase, MetricScore


class DeterministicJudge:
    name = "deterministic"

    def __init__(self, similarity_threshold: float = 0.30) -> None:
        self.similarity_threshold = similarity_threshold

    def _similarity_for(self, case: EvalCase) -> TfidfSimilarity:
        corpus = [case.question, case.answer, *case.context_texts]
        if case.ground_truth:
            corpus.append(case.ground_truth)
        return TfidfSimilarity(corpus)

    def score(self, case: EvalCase, metric: str) -> MetricScore:
        if metric not in GENERATION_METRICS:
            raise ValueError(f"unknown generation metric: {metric}")
        sim = self._similarity_for(case)
        if metric == "faithfulness":
            return self._faithfulness(case, sim)
        if metric == "answer_relevance":
            return self._answer_relevance(case, sim)
        return self._answer_correctness(case, sim)

    def _faithfulness(self, case: EvalCase, sim: TfidfSimilarity) -> MetricScore:
        """Fraction of answer claims supported by at least one context."""
        claims = split_sentences(case.answer)
        contexts = case.context_texts
        if not claims:
            return MetricScore(name="faithfulness", score=0.0, reason="empty answer")
        if not contexts:
            return MetricScore(name="faithfulness", score=0.0, reason="no contexts")
        supported = 0
        for claim in claims:
            if sim.max_similarity(claim, contexts) >= self.similarity_threshold:
                supported += 1
        score = supported / len(claims)
        return MetricScore(
            name="faithfulness",
            score=score,
            reason=f"{supported}/{len(claims)} claims supported by context",
        )

    def _answer_relevance(self, case: EvalCase, sim: TfidfSimilarity) -> MetricScore:
        score = sim.similarity(case.answer, case.question)
        return MetricScore(
            name="answer_relevance",
            score=score,
            reason="lexical similarity of answer to question",
        )

    def _answer_correctness(self, case: EvalCase, sim: TfidfSimilarity) -> MetricScore:
        if not case.ground_truth:
            raise ValueError("answer_correctness requires ground_truth")
        f1 = token_f1(case.answer, case.ground_truth)
        cos = sim.similarity(case.answer, case.ground_truth)
        score = mean([f1, cos])
        return MetricScore(
            name="answer_correctness",
            score=score,
            reason=f"token_f1={f1:.2f}, cosine={cos:.2f}",
        )
