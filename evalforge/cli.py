"""EvalForge command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from evalforge.calibrate import calibrate
from evalforge.gate import evaluate_gate
from evalforge.judges import build_judge
from evalforge.report import render_console, save_json, to_markdown
from evalforge.runner import EvalRunner
from evalforge.suites.loader import load_suite

app = typer.Typer(help="EvalForge — evaluation harness for RAG/LLM systems.", add_completion=False)
console = Console()


@app.command()
def run(
    suite: str = typer.Argument(..., help="Path to a suite (.yaml/.json)"),
    judge: str = typer.Option("deterministic", help="deterministic | anthropic"),
    out: str | None = typer.Option(None, help="Write the JSON report here"),
    markdown: str | None = typer.Option(None, help="Write a Markdown report here"),
    otel: bool = typer.Option(False, help="Emit OpenTelemetry spans to the console"),
) -> None:
    """Run a suite and print the aggregate scores."""
    report = EvalRunner(build_judge(judge), otel_enabled=otel).run(load_suite(suite))
    render_console(report, console)
    if out:
        console.print(f"[dim]wrote {save_json(report, out)}[/dim]")
    if markdown:
        Path(markdown).write_text(to_markdown(report))
        console.print(f"[dim]wrote {markdown}[/dim]")


@app.command()
def gate(
    suite: str = typer.Argument(...),
    judge: str = typer.Option("deterministic"),
    min_faithfulness: float = typer.Option(0.0),
    min_answer_relevance: float = typer.Option(0.0),
    min_answer_correctness: float = typer.Option(0.0),
    min_context_precision: float = typer.Option(0.0),
    min_hit_rate: float = typer.Option(0.0),
) -> None:
    """Fail (exit 1) if any metric falls below its threshold. For CI."""
    thresholds = {
        "faithfulness": min_faithfulness,
        "answer_relevance": min_answer_relevance,
        "answer_correctness": min_answer_correctness,
        "context_precision": min_context_precision,
        "hit_rate": min_hit_rate,
    }
    thresholds = {k: v for k, v in thresholds.items() if v > 0}
    report = EvalRunner(build_judge(judge)).run(load_suite(suite))
    render_console(report, console)
    result = evaluate_gate(report, thresholds)
    if result.passed:
        console.print("[bold green]GATE PASSED[/bold green]")
        raise typer.Exit(0)
    console.print("[bold red]GATE FAILED[/bold red]")
    for failure in result.failures:
        console.print(f"  [red]✗[/red] {failure.reason}")
    raise typer.Exit(1)


@app.command(name="calibrate")
def calibrate_command(
    suite: str = typer.Argument(...),
    labels: str = typer.Option(..., help="JSON of {case_id: {metric: human_score}}"),
    judge: str = typer.Option("deterministic"),
) -> None:
    """Compare the judge against human labels (correlation + MAE)."""
    human = json.loads(Path(labels).read_text())
    report = EvalRunner(build_judge(judge)).run(load_suite(suite))
    results = calibrate(report, human)
    if not results:
        console.print("[yellow]No overlapping case/metric labels to calibrate against.[/yellow]")
        raise typer.Exit(0)
    for c in results:
        console.print(
            f"[bold]{c.metric}[/bold] (n={c.n}): "
            f"pearson={c.pearson:.3f}  spearman={c.spearman:.3f}  mae={c.mae:.3f}"
        )


if __name__ == "__main__":
    app()
