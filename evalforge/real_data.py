"""Load the real RAGTruth hallucination benchmark and map it onto EvalForge.

RAGTruth is a word-level hallucination benchmark for retrieval-augmented
generation: for each ``(query, context, output)`` triple, human annotators
marked every span of the LLM ``output`` that is **not supported by** or
**conflicts with** the retrieved ``context``. We turn those span annotations
into a per-row binary *faithfulness* label — a response with any hallucination
annotation is **unfaithful** (0), one with none is **faithful** (1) — and use
them as ground truth to validate EvalForge's own faithfulness judge against
real human judgments.

Two layers live here:

* :func:`fetch_ragtruth` pages the Hugging Face datasets-server rows API,
  derives the label, truncates long fields, selects a class-balanced and
  task-stratified subset, and writes a vendored JSONL snapshot. It is run once
  to (re)generate ``datasets/ragtruth.jsonl``; it is never needed at eval time.
* :func:`load_ragtruth` reads that vendored snapshot, and :func:`to_eval_case`
  maps a record onto the project's :class:`~evalforge.schemas.EvalCase` so the
  real judge and metrics score it with no special-casing.

Source: RAGTruth (Niu et al., 2024), redistributed as ``wandb/RAGTruth-processed``
on the Hugging Face Hub. See ``datasets/README.md`` for provenance and license.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

from evalforge.metrics.similarity import split_sentences
from evalforge.schemas import Context, EvalCase, Suite

DATASET = "wandb/RAGTruth-processed"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
_USER_AGENT = "evalforge-real-benchmark/1.0"

DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
DEFAULT_PATH = DATASETS_DIR / "ragtruth.jsonl"

FAITHFUL = 1
UNFAITHFUL = 0


class RagTruthRecord(BaseModel):
    """One vendored RAGTruth row with its derived faithfulness label.

    ``label`` is 1 (faithful) or 0 (unfaithful). ``n_spans`` and the two
    processed counts are kept as provenance so the derivation can be re-checked
    against the raw human annotation at any time.
    """

    id: str
    query: str
    context: str
    output: str
    task_type: str
    label: int
    n_spans: int
    evident_conflict: int
    baseless_info: int

    @property
    def is_faithful(self) -> bool:
        return self.label == FAITHFUL


def _parse_spans(raw: object) -> list:
    """Parse the ``hallucination_labels`` field into a list of span objects."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []
    return []


def _processed_counts(row: Mapping) -> tuple[int, int]:
    processed = row.get("hallucination_labels_processed") or {}
    evident = int(processed.get("evident_conflict") or 0)
    baseless = int(processed.get("baseless_info") or 0)
    return evident, baseless


def derive_faithfulness_label(row: Mapping) -> int:
    """Derive the binary faithfulness label from the human annotations.

    A response is **unfaithful** (0) if it carries *any* hallucination
    annotation — either a marked span in ``hallucination_labels`` or a non-zero
    processed count — and **faithful** (1) only when every annotation says it is
    clean. The two encodings agree on every row of the published dataset; taking
    the union is robust even if one field were missing.
    """
    spans = _parse_spans(row.get("hallucination_labels"))
    evident, baseless = _processed_counts(row)
    has_hallucination = len(spans) > 0 or (evident + baseless) > 0
    return UNFAITHFUL if has_hallucination else FAITHFUL


def _record_from_raw(
    row: Mapping, *, context_chars: int, output_chars: int, query_chars: int
) -> RagTruthRecord:
    spans = _parse_spans(row.get("hallucination_labels"))
    evident, baseless = _processed_counts(row)
    return RagTruthRecord(
        id=str(row["id"]),
        query=(row.get("query") or "")[:query_chars],
        context=(row.get("context") or "")[:context_chars],
        output=(row.get("output") or "")[:output_chars],
        task_type=str(row.get("task_type") or "unknown"),
        label=derive_faithfulness_label(row),
        n_spans=len(spans),
        evident_conflict=evident,
        baseless_info=baseless,
    )


# ---------------------------------------------------------------------------
# Network layer (monkeypatchable)
# ---------------------------------------------------------------------------

def _default_urlopen(url: str, timeout: float = 90.0) -> bytes:
    """Fetch ``url`` and return the raw response body.

    This is the single network seam: tests and callers replace
    ``evalforge.real_data.urlopen`` with a fake to run the fetch path offline.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (trusted host)
        return response.read()


#: Module-level network seam. Reassign in tests to avoid real HTTP.
urlopen = _default_urlopen


def _page_url(offset: int, length: int) -> str:
    return (
        f"{ROWS_API}?dataset=wandb%2FRAGTruth-processed"
        f"&config=default&split=train&offset={offset}&length={length}"
    )


def _fetch_page(
    offset: int,
    length: int = PAGE_SIZE,
    *,
    timeout: float = 90.0,
    retries: int = 6,
    backoff: float = 5.0,
    sleep=time.sleep,
) -> dict:
    """Fetch one page of rows with bounded exponential backoff on failure.

    The datasets-server rate-limits (HTTP 429); each retry waits
    ``backoff * 2**attempt`` seconds (capped at 60) before trying again.
    """
    url = _page_url(offset, length)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return json.loads(urlopen(url, timeout))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as error:
            last_error = error
            if attempt == retries:
                break
            sleep(min(60.0, backoff * (2**attempt)))
    raise RuntimeError(f"failed to fetch rows at offset {offset}: {last_error}")


def _iter_raw_rows(
    *,
    timeout: float = 90.0,
    retries: int = 6,
    backoff: float = 5.0,
    page_delay: float = 1.3,
    sleep=time.sleep,
    max_rows: int | None = None,
) -> Iterator[Mapping]:
    """Yield raw row dicts across the whole dataset, one page at a time."""
    first = _fetch_page(
        0, PAGE_SIZE, timeout=timeout, retries=retries, backoff=backoff, sleep=sleep
    )
    total = int(first.get("num_rows_total") or 0)
    if max_rows is not None:
        total = min(total, max_rows)
    yielded = 0
    page = first
    offset = 0
    while offset < total:
        for item in page["rows"]:
            yield item["row"]
            yielded += 1
            if max_rows is not None and yielded >= max_rows:
                return
        offset += PAGE_SIZE
        if offset >= total:
            break
        sleep(page_delay)
        page = _fetch_page(
            offset, PAGE_SIZE, timeout=timeout, retries=retries, backoff=backoff, sleep=sleep
        )


def _balanced_stratified(
    candidates: Sequence[RagTruthRecord], per_class: int
) -> list[RagTruthRecord]:
    """Pick ``per_class`` faithful and ``per_class`` unfaithful records.

    Within each class the picks are round-robined across task types (sorted by
    id for determinism) so every task type that has rows of that class is
    represented, rather than a single task dominating.
    """
    selected: list[RagTruthRecord] = []
    for label in (FAITHFUL, UNFAITHFUL):
        by_task: dict[str, list[RagTruthRecord]] = defaultdict(list)
        for record in candidates:
            if record.label == label:
                by_task[record.task_type].append(record)
        for task in by_task:
            by_task[task].sort(key=lambda r: r.id)
        queues = [by_task[task] for task in sorted(by_task)]
        picked: list[RagTruthRecord] = []
        cursors = [0] * len(queues)
        while len(picked) < per_class and any(
            cursors[i] < len(queues[i]) for i in range(len(queues))
        ):
            for i, queue in enumerate(queues):
                if cursors[i] < len(queue):
                    picked.append(queue[cursors[i]])
                    cursors[i] += 1
                    if len(picked) >= per_class:
                        break
        selected.extend(picked)
    selected.sort(key=lambda r: (r.task_type, r.id))
    return selected


def write_jsonl(records: Iterable[RagTruthRecord], path: str | Path) -> Path:
    """Write records to ``path`` as one compact JSON object per line."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    return out


def fetch_ragtruth(
    *,
    per_class: int = 500,
    quality: str = "good",
    context_chars: int = 2000,
    output_chars: int = 1200,
    query_chars: int = 1000,
    path: str | Path | None = DEFAULT_PATH,
    timeout: float = 90.0,
    retries: int = 6,
    backoff: float = 5.0,
    page_delay: float = 1.3,
    sleep=time.sleep,
    max_rows: int | None = None,
) -> list[RagTruthRecord]:
    """Fetch, label, balance, and (optionally) vendor a RAGTruth subset.

    Pages the datasets-server rows API through the monkeypatchable
    :data:`urlopen` seam, keeps only rows whose ``quality`` matches, truncates
    long fields, derives the faithfulness label, then selects ``per_class``
    faithful and ``per_class`` unfaithful rows stratified across task types.
    Writes the result to ``path`` (JSONL) unless ``path`` is ``None`` and
    returns the selected records.
    """
    candidates: list[RagTruthRecord] = []
    for raw in _iter_raw_rows(
        timeout=timeout,
        retries=retries,
        backoff=backoff,
        page_delay=page_delay,
        sleep=sleep,
        max_rows=max_rows,
    ):
        if quality is not None and (raw.get("quality") or "") != quality:
            continue
        if not (raw.get("context") or "").strip() or not (raw.get("output") or "").strip():
            continue
        candidates.append(
            _record_from_raw(
                raw,
                context_chars=context_chars,
                output_chars=output_chars,
                query_chars=query_chars,
            )
        )
    selected = _balanced_stratified(candidates, per_class)
    if path is not None:
        write_jsonl(selected, path)
    return selected


# ---------------------------------------------------------------------------
# Load + map onto EvalForge
# ---------------------------------------------------------------------------

def load_ragtruth(path: str | Path = DEFAULT_PATH) -> list[RagTruthRecord]:
    """Load the vendored RAGTruth snapshot into validated records."""
    file = Path(path)
    records: list[RagTruthRecord] = []
    with file.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(RagTruthRecord.model_validate_json(line))
    return records


def to_eval_case(record: RagTruthRecord) -> EvalCase:
    """Map a RAGTruth record onto an :class:`EvalCase` for the real judge.

    The source ``context`` is split into sentence-level passages (reusing the
    project's :func:`split_sentences`), each a retrieved :class:`Context`, so
    the faithfulness judge checks every answer claim against the best-matching
    source sentence — exactly what RAGTruth annotates at the span level. No
    ground truth is attached (RAGTruth has none), so ``answer_correctness`` is
    simply not scored.
    """
    passages = split_sentences(record.context) or [record.context]
    return EvalCase(
        id=record.id,
        question=record.query,
        answer=record.output,
        contexts=[Context(text=passage, relevant=True) for passage in passages],
    )


def to_suite(records: Sequence[RagTruthRecord], name: str = "ragtruth-real") -> Suite:
    """Build an EvalForge :class:`Suite` from RAGTruth records."""
    return Suite(name=name, cases=[to_eval_case(record) for record in records])


def human_faithfulness_labels(records: Sequence[RagTruthRecord]) -> dict[str, dict[str, float]]:
    """Return ``{case_id: {"faithfulness": label}}`` for :func:`~evalforge.calibrate.calibrate`."""
    return {record.id: {"faithfulness": float(record.label)} for record in records}
