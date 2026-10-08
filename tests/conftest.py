from __future__ import annotations

from pathlib import Path

import pytest

from evalforge.schemas import Context, EvalCase
from evalforge.suites.loader import load_suite

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def clean_case() -> EvalCase:
    return EvalCase(
        id="t-clean",
        question="What is the capital of France?",
        answer="The capital of France is Paris.",
        ground_truth="Paris is the capital of France.",
        total_relevant=1,
        contexts=[
            Context(text="Paris is the capital of France.", relevant=True),
            Context(text="Bananas are a good source of potassium.", relevant=False),
        ],
    )


@pytest.fixture
def example_suite():
    return load_suite(EXAMPLES / "rag_suite.yaml")


@pytest.fixture
def example_suite_path() -> Path:
    return EXAMPLES / "rag_suite.yaml"


@pytest.fixture
def human_labels_path() -> Path:
    return EXAMPLES / "human_labels.json"
