"""Retrieval metrics computed from the per-context relevance labels.

These are label-based and deterministic — no judge or model involved — so they
are the ground truth against which a retriever is measured.
"""

from __future__ import annotations

from evalforge.schemas import EvalCase


def hit_rate(case: EvalCase) -> float:
    """1.0 if at least one retrieved context is relevant, else 0.0."""
    return 1.0 if any(c.relevant for c in case.contexts) else 0.0


def context_precision(case: EvalCase) -> float:
    """Fraction of retrieved contexts that are relevant (precision@k)."""
    if not case.contexts:
        return 0.0
    relevant = sum(1 for c in case.contexts if c.relevant)
    return relevant / len(case.contexts)


def context_recall(case: EvalCase) -> float:
    """Recall of relevant contexts.

    Against ``total_relevant`` when the case provides it (how many relevant
    passages exist in the corpus); otherwise 1.0 when any relevant context was
    retrieved, 0.0 when none was (degenerate but well-defined).
    """
    retrieved_relevant = sum(1 for c in case.contexts if c.relevant)
    if case.total_relevant and case.total_relevant > 0:
        return min(1.0, retrieved_relevant / case.total_relevant)
    return 1.0 if retrieved_relevant > 0 else 0.0


def mrr(case: EvalCase) -> float:
    """Mean reciprocal rank: 1/rank of the first relevant context."""
    for index, context in enumerate(case.contexts, start=1):
        if context.relevant:
            return 1.0 / index
    return 0.0
