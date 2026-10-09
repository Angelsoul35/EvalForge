# ADR-0003: Use a deterministic judge by default

* Status: accepted
* Date: 2026-01-21

## Context

An LLM judge gives the best quality scores, but it costs money, needs an API key and a
network, and gives slightly different answers each time. Tests and CI need something
free, offline, and repeatable.

## Decision

The default judge is deterministic. It uses lexical methods (TF-IDF cosine similarity
and token F1). The LLM judge is available behind the same interface for cases that need real
semantic judgement. It is provider-agnostic (a cascade over Anthropic, OpenAI, Gemini,
DeepSeek, or a local Ollama model), and you choose it with one flag.

## Consequences

* **Good:** the whole test suite runs offline with no key; results do not change between
  runs; new users can try the tool at once.
* **Cost:** the lexical judge can be fooled by paraphrase or by a fluent wrong answer.
  We state this plainly in the README, and `calibrate` measures it so you know when
  to switch judges.
