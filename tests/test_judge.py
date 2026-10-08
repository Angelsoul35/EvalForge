from __future__ import annotations

import pytest

from evalforge.judges.deterministic import DeterministicJudge
from evalforge.schemas import Context, EvalCase


def test_faithfulness_high_when_grounded(clean_case):
    judge = DeterministicJudge()
    assert judge.score(clean_case, "faithfulness").score >= 0.8


def test_faithfulness_low_when_unsupported():
    judge = DeterministicJudge()
    case = EvalCase(
        id="h",
        question="What is the capital of Japan?",
        answer="The sun is a star made of hydrogen and helium plasma.",
        contexts=[Context(text="Tokyo is the capital of Japan.", relevant=True)],
    )
    assert judge.score(case, "faithfulness").score < 0.5


def test_answer_relevance_orders_on_vs_off_topic(clean_case):
    judge = DeterministicJudge()
    on_topic = judge.score(clean_case, "answer_relevance").score
    off_topic_case = clean_case.model_copy(update={"answer": "I like to drink coffee on weekends."})
    off_topic = judge.score(off_topic_case, "answer_relevance").score
    assert on_topic > off_topic


def test_answer_correctness_requires_ground_truth(clean_case):
    judge = DeterministicJudge()
    assert 0.0 <= judge.score(clean_case, "answer_correctness").score <= 1.0
    no_gt = clean_case.model_copy(update={"ground_truth": None})
    with pytest.raises(ValueError):
        judge.score(no_gt, "answer_correctness")


def test_unknown_metric_raises(clean_case):
    with pytest.raises(ValueError):
        DeterministicJudge().score(clean_case, "made_up_metric")


def test_direction_on_metric_score(clean_case):
    score = DeterministicJudge().score(clean_case, "faithfulness")
    assert score.name == "faithfulness"
    assert 0.0 <= score.score <= 1.0
    assert score.reason
