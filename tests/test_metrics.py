from __future__ import annotations

from evalforge.metrics import retrieval
from evalforge.metrics.similarity import TfidfSimilarity, split_sentences, token_f1
from evalforge.schemas import Context, EvalCase


def _case(relevances: list[bool], total_relevant=None) -> EvalCase:
    return EvalCase(
        id="x",
        question="q",
        answer="a",
        total_relevant=total_relevant,
        contexts=[Context(text=f"ctx{i}", relevant=r) for i, r in enumerate(relevances)],
    )


def test_hit_rate():
    assert retrieval.hit_rate(_case([False, True, False])) == 1.0
    assert retrieval.hit_rate(_case([False, False])) == 0.0


def test_context_precision():
    assert retrieval.context_precision(_case([True, True, False, False])) == 0.5
    assert retrieval.context_precision(_case([False])) == 0.0


def test_context_recall_with_total():
    assert retrieval.context_recall(_case([True, False], total_relevant=4)) == 0.25
    assert retrieval.context_recall(_case([True, True], total_relevant=2)) == 1.0


def test_mrr():
    assert retrieval.mrr(_case([False, True, True])) == 0.5
    assert retrieval.mrr(_case([True])) == 1.0
    assert retrieval.mrr(_case([False, False])) == 0.0


def test_similarity_extremes():
    corpus = ["the cat sat on the mat", "a dog ran in the park", "the cat sat on the mat"]
    sim = TfidfSimilarity(corpus)
    assert sim.similarity("the cat sat on the mat", "the cat sat on the mat") > 0.99
    assert sim.similarity("the cat sat on the mat", "a dog ran in the park") < 0.3


def test_similarity_empty_is_zero():
    assert TfidfSimilarity([]).similarity("a", "b") == 0.0


def test_token_f1():
    assert token_f1("the capital is paris", "the capital is paris") == 1.0
    assert token_f1("completely different words here", "paris france capital city") == 0.0
    assert 0.0 < token_f1("the capital of france is paris", "paris is the capital") < 1.0


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("") == []
