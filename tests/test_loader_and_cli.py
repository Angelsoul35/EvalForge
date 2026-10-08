from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from evalforge.cli import app
from evalforge.schemas import Suite
from evalforge.suites.loader import load_suite

runner = CliRunner()


def test_load_yaml_suite(example_suite_path):
    suite = load_suite(example_suite_path)
    assert isinstance(suite, Suite)
    assert len(suite.cases) == 5


def test_load_json_suite(tmp_path, example_suite_path):
    suite = load_suite(example_suite_path)
    json_path = tmp_path / "suite.json"
    json_path.write_text(suite.model_dump_json())
    reloaded = load_suite(json_path)
    assert len(reloaded.cases) == len(suite.cases)


def test_unsupported_suffix_raises(tmp_path):
    bad = tmp_path / "suite.txt"
    bad.write_text("nope")
    with pytest.raises(ValueError):
        load_suite(bad)


def test_duplicate_case_ids_rejected():
    with pytest.raises(ValueError):
        Suite.model_validate(
            {
                "name": "dup",
                "cases": [
                    {"id": "a", "question": "q", "answer": "x"},
                    {"id": "a", "question": "q", "answer": "y"},
                ],
            }
        )


def test_cli_run(example_suite_path, tmp_path):
    out = tmp_path / "report.json"
    result = runner.invoke(app, ["run", str(example_suite_path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "faithfulness" in json.loads(out.read_text())["aggregates"]


def test_cli_gate_fails_on_impossible_threshold(example_suite_path):
    result = runner.invoke(app, ["gate", str(example_suite_path), "--min-faithfulness", "1.01"])
    assert result.exit_code == 1
    assert "GATE FAILED" in result.output


def test_cli_gate_passes_on_trivial_threshold(example_suite_path):
    result = runner.invoke(app, ["gate", str(example_suite_path), "--min-hit-rate", "0.01"])
    assert result.exit_code == 0
    assert "GATE PASSED" in result.output


def test_cli_calibrate(example_suite_path, human_labels_path):
    result = runner.invoke(
        app, ["calibrate", str(example_suite_path), "--labels", str(human_labels_path)]
    )
    assert result.exit_code == 0
    assert "faithfulness" in result.output
