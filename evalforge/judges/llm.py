"""An LLM-as-judge that works with any provider through the LLM cascade.

The judge does not know or care which provider answers. It takes any
``LLMClient`` — a single provider or a fallback cascade over several — builds a
rubric prompt, asks for a JSON verdict, and parses it. Calibrate it against
human labels before you trust it (``evalforge calibrate``).

Install a provider's needs and set its API key, then select this judge with
``--judge llm``. The provider order is configured with ``EF_LLM_PROVIDERS``.
"""

from __future__ import annotations

import json

from evalforge.judges.base import GENERATION_METRICS
from evalforge.llm.base import LLMClient
from evalforge.llm.factory import build_llm
from evalforge.schemas import EvalCase, MetricScore

_RUBRICS = {
    "faithfulness": (
        "Rate how well the ANSWER is supported by the CONTEXTS. 1.0 means every claim is "
        "grounded in the contexts; 0.0 means the answer is unsupported or contradicts them."
    ),
    "answer_relevance": (
        "Rate how directly the ANSWER addresses the QUESTION. 1.0 means fully on point; "
        "0.0 means off-topic."
    ),
    "answer_correctness": (
        "Rate how correct the ANSWER is versus the GROUND TRUTH. 1.0 means equivalent in "
        "meaning; 0.0 means wrong."
    ),
}

_SYSTEM = (
    "You are a strict evaluation judge for a retrieval-augmented generation system. "
    "You return only a JSON object of the form "
    '{"score": <float 0..1>, "reason": "<one sentence>"}. No other text.'
)


class LLMJudge:
    name = "llm"

    def __init__(self, llm: LLMClient | None = None, provider_spec: str | None = None) -> None:
        if llm is not None:
            self.llm = llm
        elif provider_spec:
            self.llm = build_llm(provider_spec)
        else:
            self.llm = build_llm()

    def score(self, case: EvalCase, metric: str) -> MetricScore:
        if metric not in GENERATION_METRICS:
            raise ValueError(f"unknown generation metric: {metric}")
        if metric == "answer_correctness" and not case.ground_truth:
            raise ValueError("answer_correctness requires ground_truth")

        result = self.llm.complete(_SYSTEM, self._prompt(case, metric), max_tokens=256)
        score, reason = _parse_verdict(result.text)
        return MetricScore(name=metric, score=score, reason=f"[{result.provider}] {reason}")

    def _prompt(self, case: EvalCase, metric: str) -> str:
        contexts = "\n".join(f"- {c.text}" for c in case.contexts) or "(none)"
        parts = [
            _RUBRICS[metric],
            "",
            f"QUESTION:\n{case.question}",
            f"CONTEXTS:\n{contexts}",
            f"ANSWER:\n{case.answer}",
        ]
        if metric == "answer_correctness":
            parts.append(f"GROUND TRUTH:\n{case.ground_truth}")
        parts.append('\nReturn only JSON: {"score": <float>, "reason": "<one sentence>"}')
        return "\n".join(parts)


def _parse_verdict(text: str) -> tuple[float, str]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return 0.0, "unparseable judge response"
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return 0.0, "unparseable judge response"
    score = max(0.0, min(1.0, float(payload.get("score", 0.0))))
    return score, str(payload.get("reason", ""))
