from __future__ import annotations

import json

from evalforge.calibrate import calibrate
from evalforge.gate import evaluate_gate
from evalforge.judges.deterministic import DeterministicJudge
from evalforge.runner import EvalRunner


def test_runner_produces_all_metrics(example_suite):
    report = EvalRunner(DeterministicJudge()).run(example_suite)
    assert report.n_cases == 5
    for metric in [
        "hit_rate",
        "context_precision",
        "faithfulness",
        "answer_relevance",
        "answer_correctness",
    ]:
        assert metric in report.aggregates


def test_runner_reflects_case_quality(example_suite):
    report = EvalRunner(DeterministicJudge()).run(example_suite)
    by_id = {r.case_id: r for r in report.results}
    # The off-topic case should score far lower on relevance than a clean case.
    assert by_id["c1-clean"].value("answer_relevance") > by_id["c3-off-topic"].value(
        "answer_relevance"
    )
    # The retrieval-miss case retrieved no relevant context.
    assert by_id["c4-retrieval-miss"].value("hit_rate") == 0.0
    assert by_id["c1-clean"].value("hit_rate") == 1.0


def test_gate_pass_and_fail(example_suite):
    report = EvalRunner(DeterministicJudge()).run(example_suite)
    assert evaluate_gate(report, {"hit_rate": 0.0}).passed is True
    failing = evaluate_gate(report, {"faithfulness": 1.01})
    assert failing.passed is False
    assert failing.failures[0].metric == "faithfulness"


def test_gate_missing_metric_fails(example_suite):
    report = EvalRunner(DeterministicJudge()).run(example_suite)
    result = evaluate_gate(report, {"nonexistent_metric": 0.5})
    assert result.passed is False
    assert result.failures[0].actual is None


def test_calibration_against_human_labels(example_suite, human_labels_path):
    report = EvalRunner(DeterministicJudge()).run(example_suite)
    labels = json.loads(human_labels_path.read_text())
    results = calibrate(report, labels)
    assert {c.metric for c in results} == {"faithfulness", "answer_relevance", "answer_correctness"}
    for c in results:
        assert c.n == 5
        assert c.mae >= 0.0
