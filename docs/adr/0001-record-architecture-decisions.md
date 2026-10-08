# ADR-0001: Record architecture decisions

* Status: accepted
* Date: 2026-01-20

## Context

EvalForge makes a few design choices that are not obvious (a judge interface, a lexical
default judge, a calibration step). A future reader needs to know *why*, not just *what*.

## Decision

We keep short Architecture Decision Records (ADRs) in `docs/adr/`, one per important
choice. Each one has three parts: Context, Decision, and Consequences.

## Consequences

* Choices are easy to review later.
* The cost is small: one short Markdown file per decision.
