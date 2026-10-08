# ADR-0002: Score retrieval and generation separately

* Status: accepted
* Date: 2026-01-20

## Context

When a RAG answer is bad, there are two possible causes: the retriever brought back the
wrong context, or the model wrote a bad answer from good context. A single score mixes
these together and hides the real problem.

## Decision

We compute retrieval metrics (`hit_rate`, `context_precision`, `context_recall`, `mrr`)
from relevance labels, and generation metrics (`faithfulness`, `answer_relevance`,
`answer_correctness`) from the judge. We report them separately.

## Consequences

* **Good:** you can tell which part of the system broke. A good answer with
  `hit_rate = 0` means the retriever failed but the model still knew the answer.
* **Cost:** suites must include relevance labels for the contexts. This is extra work
  when building a suite, but it is the information that makes the metrics useful.
