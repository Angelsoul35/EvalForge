from __future__ import annotations

from evalforge.judges.deterministic import DeterministicJudge
from evalforge.report import load_json, render_console, save_json, to_markdown
from evalforge.runner import EvalRunner
from evalforge.tracing import span


def _report(example_suite, otel=False):
    return EvalRunner(DeterministicJudge(), otel_enabled=otel).run(example_suite)


def test_markdown_contains_metrics_and_cases(example_suite):
    md = to_markdown(_report(example_suite))
    assert "# Eval report" in md
    assert "faithfulness" in md
    assert "c1-clean" in md


def test_json_roundtrip(example_suite, tmp_path):
    report = _report(example_suite)
    path = save_json(report, tmp_path / "r.json")
    reloaded = load_json(path)
    assert reloaded.suite_name == report.suite_name
    assert reloaded.aggregates == report.aggregates


def test_render_console_runs(example_suite, capsys):
    render_console(_report(example_suite))
    captured = capsys.readouterr()
    assert "EvalForge" in captured.out


def test_tracing_span_enabled_and_disabled(example_suite):
    # Disabled is a no-op; enabled must not raise and must produce a valid report.
    with span("unit", False, k="v"):
        pass
    report = _report(example_suite, otel=True)
    assert report.n_cases == 5
