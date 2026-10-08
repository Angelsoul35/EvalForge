# ADR-0004: Always offer judge calibration

* Status: accepted
* Date: 2026-01-21

## Context

An automated judge that nobody has checked against people is a risk. If it disagrees
with human opinion, its scores are noise, and a CI gate built on them is worse than no
gate.

## Decision

EvalForge ships a `calibrate` command. Given human scores for the same cases, it reports
Pearson and Spearman correlation and mean absolute error, per metric.

## Consequences

* **Good:** you can prove, with numbers, whether the judge is trustworthy for each
  metric before you rely on it.
* **Cost:** you need some human-labelled cases. A small set is enough to get a useful
  signal, and you only need to do it when the judge or the prompt changes.
