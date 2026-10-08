from __future__ import annotations

import pytest

from evalforge.judges import build_judge
from evalforge.judges.llm import LLMJudge
from evalforge.llm.base import LLMError
from evalforge.llm.cascade import CascadeLLM
from evalforge.llm.factory import build_llm
from evalforge.llm.fake import FailingLLM, FakeLLM
from evalforge.llm.providers import OpenAIClient, build_client


def test_cascade_falls_back_to_next_provider():
    cascade = CascadeLLM([FailingLLM("a"), FailingLLM("b"), FakeLLM("recovered")])
    result = cascade.complete("sys", "prompt")
    assert result.text == "recovered"
    assert result.provider == "fake"


def test_cascade_raises_when_all_fail():
    cascade = CascadeLLM([FailingLLM("a"), FailingLLM("b")])
    with pytest.raises(LLMError) as exc:
        cascade.complete("sys", "prompt")
    assert "all providers failed" in str(exc.value)


def test_cascade_requires_at_least_one_client():
    with pytest.raises(ValueError):
        CascadeLLM([])


def test_cascade_model_lists_providers():
    cascade = CascadeLLM([FakeLLM(), FailingLLM("x")])
    assert cascade.model == "cascade[fake,x]"


def test_build_llm_single_vs_cascade():
    single = build_llm("openai")
    assert single.name == "openai"
    multi = build_llm("anthropic,openai,ollama")
    assert isinstance(multi, CascadeLLM)
    assert [c.name for c in multi.clients] == ["anthropic", "openai", "ollama"]


def test_build_client_unknown_provider():
    with pytest.raises(ValueError):
        build_client("not-a-provider")


def test_openai_client_without_key_raises_llm_error():
    client = OpenAIClient(api_key="")
    with pytest.raises(LLMError):
        client.complete("sys", "prompt")


def test_llm_judge_parses_verdict(clean_case):
    judge = LLMJudge(llm=FakeLLM('{"score": 0.82, "reason": "mostly grounded"}'))
    score = judge.score(clean_case, "faithfulness")
    assert score.score == 0.82
    assert "mostly grounded" in score.reason
    assert "[fake]" in score.reason


def test_llm_judge_clamps_and_handles_bad_json(clean_case):
    assert LLMJudge(llm=FakeLLM('{"score": 1.9}')).score(clean_case, "faithfulness").score == 1.0
    assert LLMJudge(llm=FakeLLM("not json")).score(clean_case, "faithfulness").score == 0.0


def test_llm_judge_requires_ground_truth(clean_case):
    judge = LLMJudge(llm=FakeLLM('{"score": 1.0}'))
    no_gt = clean_case.model_copy(update={"ground_truth": None})
    with pytest.raises(ValueError):
        judge.score(no_gt, "answer_correctness")


def test_build_judge_llm_with_injected_client(clean_case):
    judge = build_judge("llm", llm=FakeLLM('{"score": 0.5, "reason": "ok"}'))
    assert isinstance(judge, LLMJudge)
    assert judge.score(clean_case, "answer_relevance").score == 0.5
