"""Render and persist suite reports (JSON, Markdown, and a console table)."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from evalforge.schemas import SuiteReport


def save_json(report: SuiteReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report.model_dump_json(indent=2))
    return p


def load_json(path: str | Path) -> SuiteReport:
    return SuiteReport.model_validate_json(Path(path).read_text())


def to_markdown(report: SuiteReport) -> str:
    lines = [
        f"# Eval report — {report.suite_name}",
        "",
        f"- Judge: `{report.judge}`",
        f"- Cases: {report.n_cases}",
        f"- Generated: {report.created_at}",
        "",
        "## Aggregate scores",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]
    for metric, score in report.aggregates.items():
        lines.append(f"| {metric} | {score:.3f} |")
    lines += ["", "## Per-case scores", ""]
    metrics = sorted({m for r in report.results for m in r.scores})
    header = "| case | " + " | ".join(metrics) + " |"
    sep = "|---|" + "|".join(["---"] * len(metrics)) + "|"
    lines += [header, sep]
    for result in report.results:
        cells = [f"{result.scores[m].score:.2f}" if m in result.scores else "—" for m in metrics]
        lines.append(f"| {result.case_id} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_console(report: SuiteReport, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title=f"EvalForge — {report.suite_name} (judge: {report.judge})")
    table.add_column("Metric", style="bold")
    table.add_column("Aggregate", justify="right")
    for metric, score in report.aggregates.items():
        style = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
        table.add_row(metric, f"[{style}]{score:.3f}[/{style}]")
    console.print(table)
