"""Judge factory."""

from __future__ import annotations

from evalforge.judges.base import GENERATION_METRICS, Judge
from evalforge.judges.deterministic import DeterministicJudge

__all__ = ["GENERATION_METRICS", "Judge", "DeterministicJudge", "build_judge"]


def build_judge(name: str, **kwargs) -> Judge:
    """Construct a judge by name.

    * ``deterministic`` — offline, lexical (the default).
    * ``llm`` — provider-agnostic LLM-as-judge over a cascade of providers
      (configure the order with ``provider_spec``, e.g. ``anthropic,openai,ollama``).
    """
    if name == "deterministic":
        return DeterministicJudge(similarity_threshold=kwargs.get("similarity_threshold", 0.30))
    if name == "llm":
        from evalforge.judges.llm import LLMJudge

        return LLMJudge(llm=kwargs.get("llm"), provider_spec=kwargs.get("provider_spec"))
    raise ValueError(f"unknown judge: {name!r} (use 'deterministic' or 'llm')")
