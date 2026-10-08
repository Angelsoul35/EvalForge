"""An LLM-as-judge backed by the Anthropic API (optional).

Used when a strong semantic judgement is worth the cost; the default
:class:`DeterministicJudge` is used otherwise. The judge model defaults to
``claude-sonnet-5-5`` — the documented choice for a secondary evaluator role —
and is configurable. This module imports ``anthropic`` lazily so the package is
only required when this judge is actually used (``pip install evalforge[anthropic]``).

The judge is calibrated against human labels like any other
(see ``evalforge calibrate``): never trust a judge you have not measured.
"""

from __future__ import annotations

import json

from evalforge.judges.base import GENERATION_METRICS
from evalforge.schemas import EvalCase, MetricScore

_DEFAULT_MODEL = "claude-sonnet-5-5"

_RUBRICS = {
    "faithfulness": (
        "Rate how well the ANSWER is supported by the CONTEXTS. 1.0 means every "
        "claim is grounded in the contexts; 0.0 means the answer is unsupported or "
        "contradicts them."
    ),
    "answer_relevance": (
        "Rate how directly the ANSWER addresses the QUESTION. 1.0 means fully on "
        "point; 0.0 means off-topic."
    ),
    "answer_correctness": (
        "Rate how correct the ANSWER is versus the GROUND TRUTH. 1.0 means "
        "equivalent in meaning; 0.0 means wrong."
    ),
}

_SYSTEM = (
    "You are a strict evaluation judge for a retrieval-augmented generation system. "
    "You return only a JSON object of the form "
    '{"score": <float 0..1>, "reason": "<one sentence>"}. No other text.'
)


class AnthropicJudge:
    name = "anthropic"

    def __init__(self, model: str = _DEFAULT_MODEL, api_key: str | None = None) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "The Anthropic judge requires the 'anthropic' package. "
                "Install it with: pip install 'evalforge[anthropic]'"
            ) from exc
        self.model = model
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def score(self, case: EvalCase, metric: str) -> MetricScore:
        if metric not in GENERATION_METRICS:
            raise ValueError(f"unknown generation metric: {metric}")
        if metric == "answer_correctness" and not case.ground_truth:
            raise ValueError("answer_correctness requires ground_truth")

        prompt = self._build_prompt(case, metric)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            output_config={"effort": "low"},
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":  # pragma: no cover - needs live API
            return MetricScore(name=metric, score=0.0, reason="judge refused to score")

        text = "".join(block.text for block in response.content if block.type == "text")
        score, reason = _parse_verdict(text)
        return MetricScore(name=metric, score=score, reason=reason)

    def _build_prompt(self, case: EvalCase, metric: str) -> str:
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
    """Parse the judge's JSON verdict, clamping the score to [0, 1]."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return 0.0, "unparseable judge response"
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return 0.0, "unparseable judge response"
    score = float(payload.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    return score, str(payload.get("reason", ""))
