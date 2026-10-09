# Datasets

## `ragtruth.jsonl` — real human-labelled hallucinations (RAGTruth)

A balanced 1,000-row subset (500 faithful + 500 unfaithful) of **RAGTruth**, a
word-level hallucination benchmark for retrieval-augmented generation: real LLM
responses to real (query, context) pairs, with human annotations marking hallucinated
content. Used to validate EvalForge's faithfulness judge against real human judgments
(see *Validated on real human hallucination labels* in the top-level README).

**Source.** RAGTruth (Niu et al., 2024), via the Hugging Face mirror
[`wandb/RAGTruth-processed`](https://huggingface.co/datasets/wandb/RAGTruth-processed).

**Label derivation.** Each row's binary faithfulness label is derived from the human
annotation: a response is **unfaithful (0)** if it carries *any* hallucination
annotation — a marked span in `hallucination_labels`, or a non-zero
`hallucination_labels_processed` count (`evident_conflict` / `baseless_info`) — and
**faithful (1)** only when every annotation says it is clean. The raw counts are kept in
each record (`n_spans`, `evident_conflict`, `baseless_info`) so the derivation can be
re-checked. Long contexts/outputs are truncated for size.

| Field                               | Meaning                                         |
| ----------------------------------- | ----------------------------------------------- |
| `query`, `context`, `output`        | the real RAG prompt, retrieved source, response |
| `label`                             | 1 faithful / 0 unfaithful (derived)             |
| `task_type`                         | RAGTruth task (QA, Summary, Data2txt)           |
| `n_spans`, `evident_conflict`, `baseless_info` | raw annotation provenance            |

### Regenerating

```bash
python -c "from evalforge.real_data import fetch_ragtruth; fetch_ragtruth()"
```

Needs outbound access to `datasets-server.huggingface.co`. RAGTruth's licence and the
Hugging Face mirror's terms apply. Vendored for reproducible benchmarking only. Please
cite RAGTruth (Niu et al., 2024) if you use it.
