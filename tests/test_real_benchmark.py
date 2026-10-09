"""Functional tests for the real RAGTruth validation.

They prove the vendored data is the real, balanced RAGTruth subset, that the binary
faithfulness label is derived correctly from the human annotations, that EvalForge's
deterministic judge discriminates real hallucinations above chance on it, and that the
discrimination is real signal — both controls (shuffled labels, random context) collapse
to chance.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from evalforge.real_benchmark import discriminate, format_report, run_benchmark
from evalforge.real_data import derive_faithfulness_label, load_ragtruth


@pytest.fixture(scope="module")
def records():
    return load_ragtruth()


@pytest.fixture(scope="module")
def report(records):
    # A balanced subset keeps the suite fast while still exercising the full
    # pipeline; ROC-AUC is stable at this size.
    faithful = [r for r in records if r.is_faithful][:200]
    unfaithful = [r for r in records if not r.is_faithful][:200]
    return run_benchmark(faithful + unfaithful)


def test_loads_the_real_balanced_ragtruth(records):
    assert len(records) == 1000
    faithful = [r for r in records if r.is_faithful]
    unfaithful = [r for r in records if not r.is_faithful]
    assert len(faithful) == 500 and len(unfaithful) == 500  # balanced
    for r in records:
        assert r.context and r.output and r.task_type


def test_faithfulness_label_derivation():
    # An annotated hallucination (a span, or a non-zero processed count) -> unfaithful.
    assert derive_faithfulness_label(
        {"hallucination_labels_processed": {"evident_conflict": 1, "baseless_info": 0}}
    ) == 0
    assert derive_faithfulness_label({"hallucination_labels": '[{"start": 1}]'}) == 0
    # No annotation at all -> faithful.
    assert derive_faithfulness_label(
        {"hallucination_labels": "", "hallucination_labels_processed": {}}
    ) == 1


def test_judge_discriminates_and_controls_collapse(report):
    d = report.real
    # Real discrimination is clearly above chance on the hardest hallucination set.
    assert d.roc_auc >= 0.55, d
    # ...and it is real signal: both controls collapse toward 0.5.
    assert report.auc_shuffled_labels <= 0.58, report.auc_shuffled_labels
    assert report.auc_random_context <= 0.58, report.auc_random_context
    assert d.roc_auc > report.auc_shuffled_labels + 0.02
    assert d.roc_auc > report.auc_random_context + 0.02
    # Balanced evaluation set (the subset the fixture built).
    assert d.n_faithful == d.n_unfaithful == 200


def test_discriminate_is_a_correct_metric():
    # Perfectly separable: faithful scores strictly above unfaithful -> AUC 1.0.
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    d = discriminate(labels, scores)
    assert d.roc_auc == pytest.approx(1.0)
    # Inverted scores -> AUC 0.0 (the metric genuinely reads order).
    assert discriminate(labels, 1 - scores).roc_auc == pytest.approx(0.0)


def test_calibration_is_reported(report):
    assert report.calibration is not None
    assert -1.0 <= report.calibration.pearson <= 1.0


def test_format_report_renders(report):
    text = format_report(report)
    assert "RAGTruth" in text
    assert f"{report.real.roc_auc:.3f}" in text
    assert "control" in text.lower()


def test_main_prints(capsys, report, monkeypatch):
    import evalforge.real_benchmark as rb

    # Reuse the already-computed report so this does not re-run the full benchmark.
    monkeypatch.setattr(rb, "run_benchmark", lambda *a, **k: report)
    rb.main()
    assert "RAGTruth" in capsys.readouterr().out


def test_loader_rejects_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ragtruth(tmp_path / "nope.jsonl")


def test_fetch_balances_and_derives(tmp_path, monkeypatch):
    import evalforge.real_data as rd

    def row(i, faithful, task="QA"):
        base = {
            "id": str(i), "query": f"q{i}", "context": f"ctx {i}",
            "output": f"out {i}", "task_type": task, "quality": "good",
        }
        if faithful:
            base |= {"hallucination_labels": "", "hallucination_labels_processed": {}}
        else:
            base |= {"hallucination_labels_processed": {"evident_conflict": 1, "baseless_info": 0}}
        return base

    rows = [row(i, faithful=(i % 2 == 0)) for i in range(20)]
    page = {"num_rows_total": len(rows), "rows": [{"row": r} for r in rows]}
    empty = {"num_rows_total": len(rows), "rows": []}

    def fake_urlopen(url, timeout=0.0):
        # First page returns all rows; later offsets are empty so paging stops.
        return json.dumps(page if "offset=0&" in url else empty).encode()

    monkeypatch.setattr(rd, "urlopen", fake_urlopen)
    dest = tmp_path / "rt.jsonl"
    out = rd.fetch_ragtruth(per_class=5, path=dest, page_delay=0.0, sleep=lambda *_: None)
    assert len(out) == 10  # 5 faithful + 5 unfaithful, balanced
    loaded = load_ragtruth(dest)
    assert sum(r.is_faithful for r in loaded) == 5
    assert sum(not r.is_faithful for r in loaded) == 5
