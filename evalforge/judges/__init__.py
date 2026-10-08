"""Judge factory."""

from __future__ import annotations

from evalforge.judges.base import GENERATION_METRICS, Judge
from evalforge.judges.deterministic import DeterministicJudge

__all__ = ["GENERATION_METRICS", "Judge", "DeterministicJudge", "build_judge"]


def build_judge(name: str, **kwargs) -> Judge:
    """Construct a judge by name: ``deterministic`` or ``anthropic``."""
    if name == "deterministic":
        threshold = kwargs.get("similarity_threshold", 0.30)
        return DeterministicJudge(similarity_threshold=threshold)
    if name == "anthropic":
        from evalforge.judges.anthropic import AnthropicJudge

        return AnthropicJudge(
            model=kwargs.get("model", "claude-sonnet-5-5"),
            api_key=kwargs.get("api_key"),
        )
    raise ValueError(f"unknown judge: {name!r} (use 'deterministic' or 'anthropic')")
