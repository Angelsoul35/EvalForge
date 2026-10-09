# EvalForge

**A testing and observability tool for RAG and LLM systems.**

EvalForge answers one question: "is our RAG system actually good?" It gives that
answer as numbers you can track. It scores retrieval and answer quality, uses an
LLM-as-judge that you can check against human scores, traces runs with OpenTelemetry,
and fails your CI build when quality drops.

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Why it exists

Teams build a RAG system, then change a prompt, a chunk size, or a retriever. After
that, they do not really know if the change helped or hurt. "It looks fine in the demo"
is not a quality bar.

EvalForge gives you three things:

* **Separate scores for retrieval and for generation.** This tells you *which* part
  broke. A bad answer from good context is a generation problem. A good answer from bad
  context was just luck.
* **A judge you can trust**, because you measure it against human scores before you use
  it.
* **A CI gate**, so a drop in quality fails the build instead of reaching users.

## Metrics

**Retrieval** (computed from the relevance labels — this is ground truth, no model):

| Metric | What it means |
|---|---|
| `hit_rate` | Was at least one retrieved context relevant? |
| `context_precision` | How many of the retrieved contexts are relevant |
| `context_recall` | How many of the relevant contexts were retrieved |
| `mrr` | Rank of the first relevant context (higher is better) |

**Generation** (scored by a judge):

| Metric | What it means |
|---|---|
| `faithfulness` | Do the answer's claims come from the context? (finds hallucinations) |
| `answer_relevance` | Does the answer reply to the question? |
| `answer_correctness` | Does the answer match the ground truth? |

## Judges

| Judge | When to use it |
|---|---|
| `deterministic` (default) | Offline, free, and repeatable. It uses lexical scoring (TF-IDF and token F1). Good for tests, CI, and learning. |
| `llm` | A stronger LLM-as-judge. It is **provider-agnostic**: it runs a cascade over several providers (Anthropic, OpenAI, Gemini, DeepSeek, or a local Ollama model) and falls back to the next one if one fails. Set `--providers` or `EF_LLM_PROVIDERS`. |

Both judges use the same interface, so you switch them with one flag. **Do not trust a
judge you have not measured.** `evalforge calibrate` shows how well the judge agrees
with people.

```bash
# Use the LLM judge with a provider cascade (first one that works wins)
evalforge run suites/hard.yaml --judge llm --providers anthropic,openai,ollama
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make install

# Run the example suite and see the scores
evalforge run examples/rag_suite.yaml

# Use it as a CI gate. These thresholds suit the offline deterministic judge and
# pass on the example; tighten one (e.g. --min-faithfulness 0.80) and it exits 1,
# which is how it fails a CI build when quality drops. With the LLM judge you set
# them higher — see the GitHub Actions snippet below.
evalforge gate examples/rag_suite.yaml --min-hit-rate 0.70 --min-faithfulness 0.40

# Check the judge against human scores
evalforge calibrate examples/rag_suite.yaml --labels examples/human_labels.json
```

Example output from `calibrate` (the default lexical judge against the sample labels):

```
faithfulness       (n=5): pearson=0.997  spearman=0.973  mae=0.020
answer_correctness (n=5): pearson=0.991  spearman=0.975  mae=0.319
answer_relevance   (n=5): pearson=0.732  spearman=0.359  mae=0.504
```

The lexical judge matches human faithfulness scores very well, but it is weaker on
relevance. That is exactly why calibration exists: it tells you when to trust the cheap
judge and when to use the LLM judge instead.

## A suite

A suite is a YAML or JSON file. Each case has a question, the contexts the retriever
returned (each one marked relevant or not), the system's answer, and an optional ground
truth:

```yaml
name: demo-rag-suite
cases:
  - id: c1
    question: What is the capital of France?
    answer: The capital of France is Paris.
    ground_truth: Paris is the capital of France.
    total_relevant: 1
    contexts:
      - text: Paris is the capital of France.
        relevant: true
      - text: Bananas are a good source of potassium.
        relevant: false
```

## Use it in CI

```yaml
# .github/workflows/eval.yml
- run: evalforge gate suites/production.yaml --min-faithfulness 0.80 --min-answer-correctness 0.70
```

The command prints the scores and exits with an error if any threshold is missed. So a
prompt or retriever change that drops quality fails the pull request.

## Project layout

```
evalforge/
  schemas.py     Suite / EvalCase / SuiteReport (pydantic models)
  metrics/       retrieval metrics (label-based) and lexical similarity
  judges/        the Judge interface, the deterministic judge, the LLM judge
  runner.py      runs a suite and returns a SuiteReport
  gate.py        the threshold gate for CI
  calibrate.py   judge-vs-human correlation and error
  report.py      JSON, Markdown, and console output
  tracing.py     optional OpenTelemetry spans
  cli.py         run / gate / calibrate
```

The judge is an interface (`judges/base.py`), so the two judges are easy to swap and a
new one is easy to add. See [`docs/architecture.md`](docs/architecture.md), the
[ADRs](docs/adr/), and the Spanish study guide in [`docs/estudio/`](docs/estudio/).

## Validated on real human hallucination labels (RAGTruth)

EvalForge's promise is that you don't *trust* a judge, you *measure* it against human
scores. To show that on real data, the deterministic judge is benchmarked against
**RAGTruth** — a human-annotated hallucination benchmark of real LLM responses to real
(query, context) pairs. A balanced 1,000-row subset (500 faithful / 500 unfaithful) is
vendored under `datasets/ragtruth.jsonl` (see `datasets/README.md`); the binary label is
derived from the human annotations. Run `python -m evalforge.real_benchmark`:

```
Real discrimination (faithfulness judge vs the human labels)
  ROC-AUC            : 0.598      accuracy: 0.589   F1(faithful): 0.484

Negative controls (the signal is real only if these collapse to ~0.5)
  AUC, labels shuffled : 0.518
  AUC, random context  : 0.496

Result: the free deterministic judge detects real hallucinations above chance,
and the controls confirm that discrimination is real grounding signal.
```

The point is not a big number — it is the **measurement**. On RAGTruth, the field's
hardest hallucination set (its hallucinations are subtle, in-context edits, not obvious
mismatches), the zero-cost, offline, deterministic lexical judge detects real
hallucinations **above chance** (ROC-AUC ~0.60), and the two controls prove that
discrimination is real grounding signal — it collapses to ~0.5 under shuffled labels and
under a random context. That is EvalForge doing exactly what it exists for: it tells you
precisely where your free judge stands against real humans, so you know when the lexical
judge is enough and when to `calibrate` it or switch to the LLM judge (same interface)
for a higher ceiling. Knowing the number *is* the capability. (`tests/test_real_benchmark.py`
asserts the real discrimination clears the controls and that the label derivation is
correct.)

## Testing and quality

* **40 tests** cover the metrics, the judge, the runner, the gate, calibration, the
  loader, the CLI, and the **real RAGTruth benchmark** (balanced-subset load, label
  derivation, discrimination above both controls, and the fetch path).
* CI checks coverage (at least 80%), runs `ruff` for lint, and runs `bandit` and
  `gitleaks` for security.

```bash
make test   # pytest with coverage
make lint   # ruff
make demo   # run + gate + calibrate on the example suite
```

## Limits

The default judge is **lexical**, not semantic. It is repeatable and free, but a
paraphrase or a fluent wrong answer that reuses words from the context can fool it. For
harder suites, calibrate first and then use the LLM judge. Also, EvalForge
*measures* quality; it does not fix it. Use it next to your own work on retrieval and
prompts.

## License

[MIT](LICENSE) © Angelo David Padilla Romero
