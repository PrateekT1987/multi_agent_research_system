"""Command line interface for Mars."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import ConfigError, build_settings
from .domain import DOMAIN, DOMAIN_TRACKS, SUBTOPIC_EXAMPLES
from .reliability import CallBudgetExceeded, CircuitOpenError, ProviderUnavailableError

app = typer.Typer(
    name="mars",
    help=f"Mars - a multi-agent research system for {DOMAIN} only.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _validated(settings):
    """Run startup validation and turn failures into a clean CLI exit."""
    try:
        settings.validate()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(code=2) from None
    for warning in settings.warnings():
        console.print(f"[yellow]warning:[/] {warning}")
    return settings


def _install_sigterm(pipeline) -> object:
    """Turn SIGTERM into a cooperative shutdown at the next safe boundary."""
    import signal

    def handler(signum, frame):  # pragma: no cover - signal timing
        pipeline.request_shutdown()
        console.print(
            "[yellow]Shutdown requested; stopping at the next safe boundary...[/]"
        )

    try:
        return signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):  # not the main thread / unsupported platform
        return None



@app.command()
def research(
    topic: str = typer.Option(
        "agentic AI",
        "--topic",
        "-t",
        help=f"An {DOMAIN} subtopic to research. Off-domain topics are rejected.",
    ),
    depth: str = typer.Option("standard", "--depth", "-d", help="quick | standard | deep."),
    output_dir: Path = typer.Option(
        Path("research_output"), "--output-dir", "-o", help="Where to write the report."
    ),
    rounds: int | None = typer.Option(
        None, "--rounds", "-r", help="Maximum audit/revision rounds."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Model for the editor and auditor (OpenRouter id)."
    ),
    fast_model: str | None = typer.Option(
        None, "--fast-model", help="Model for researchers and analysts (OpenRouter id)."
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the HTTP and LLM caches."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Stream progress."),
    require_approval: bool = typer.Option(
        False,
        "--require-approval/--no-require-approval",
        help="Pause for human approval at plan, draft, and revision checkpoints.",
    ),
    approval_auto_approve: bool = typer.Option(
        False,
        "--approval-auto-approve",
        help="Auto-approve all checkpoints (useful for testing).",
    ),
    approval_dir: Path = typer.Option(
        Path(".mars_approvals"), "--approval-dir", help="Directory for approval request files."
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume an interrupted run from its state journal (requires --run-id).",
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Explicit run id; also the key used for resume and replay."
    ),
) -> None:
    """Run a full multi-agent research programme and write a cited report."""
    overrides: dict[str, object] = {
        "topic": topic,
        "depth": depth,
        "output_dir": output_dir,
        "cache_enabled": not no_cache,
        "verbose": verbose,
        "require_human_approval": require_approval,
        "approval_auto_approve": approval_auto_approve,
        "approval_dir": approval_dir,
    }
    if rounds is not None:
        overrides["max_rounds"] = rounds
    if model:
        overrides["model_strong"] = model
        overrides["model_planner"] = model
    if fast_model:
        overrides["model_fast"] = fast_model
    if run_id:
        overrides["run_id"] = run_id
    if resume and not run_id:
        console.print("[red]--resume requires --run-id.[/]")
        raise typer.Exit(code=2)
    settings = build_settings(**overrides)  # type: ignore[arg-type]
    _validated(settings)
    if settings.require_human_approval and not settings.run_id:
        import uuid

        settings = settings.with_overrides(run_id=f"run-{uuid.uuid4().hex[:12]}")

    console.print(
        Panel.fit(
            f"[bold]Mars[/] research programme\n"
            f"Domain: [magenta]{DOMAIN}[/] (fixed)\n"
            f"Topic: [cyan]{settings.topic}[/]\n"
            f"Depth: {settings.depth}  |  Rounds: {settings.max_rounds}\n"
            f"Provider: {settings.llm_provider}"
            + (
                f"  |  Models: {settings.model_strong} / {settings.model_fast}"
            ),
            title="starting",
        )
    )

    from .pipeline import ResearchPipeline

    pipeline = ResearchPipeline(
        settings, reporter=lambda message: console.print(f"[dim]{message}[/]")
    )
    previous_handler = _install_sigterm(pipeline)
    try:
        result = pipeline.run(resume=resume)
    except KeyboardInterrupt:
        console.print("[red]Interrupted.[/]")
        raise typer.Exit(code=130) from None
    except (ProviderUnavailableError, CircuitOpenError, CallBudgetExceeded) as exc:
        console.print(f"[red]Run stopped:[/] {exc}")
        console.print(
            f"[dim]Run state and dead letters were kept for run {pipeline.run_id!r}. "
            "Retry later with --resume --run-id "
            f"{pipeline.run_id}, or configure another model.[/]"
        )
        raise typer.Exit(code=1) from None
    finally:
        import signal

        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
        pipeline.context.close()

    console.print()
    degraded = result.report.audit.status != "verified"
    console.print(
        Panel.fit(
            f"[bold]{result.report.title}[/]\n\n"
            f"Sources: {len(result.report.sources)}  |  "
            f"Coverage: {result.report.metrics.get('coverage', '?')} tracks  |  "
            f"Findings: {result.report.metrics.get('findings_recorded', 0)}  |  "
            f"Claims cited: {result.report.metrics.get('claims_cited', 0)}/"
            f"{result.report.metrics.get('claims_in_report', 0)}  |  "
            f"Grounding: {result.report.audit.grounded_ratio:.0%}  |  "
            f"Tokens: {result.usage.total_tokens}  |  "
            f"Cost: ~${result.usage.cost:.4f}\n"
            f"Report: {result.output_paths.get('report_md')}",
            title="complete (degraded)" if degraded else "complete",
            border_style="red" if degraded else "green",
        )
    )
    if degraded:
        console.print(
            "[yellow]This run is degraded: the audit did not verify the report. "
            "The artefact is named .DEGRADED.md and must not be consumed as a "
            "finished research result.[/]"
        )
        raise typer.Exit(code=2)


@app.command()
def doctor(
    offline: bool = typer.Option(
        False, "--offline", help="Skip the connectivity checks (no outbound requests)."
    ),
) -> None:
    """Check configuration, credentials and network access."""
    settings = build_settings()
    table = Table(title="Mars doctor", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    def add(name: str, ok: bool, detail: str) -> None:
        table.add_row(
            name,
            "[green]ok[/]" if ok else "[yellow]missing[/]",
            detail,
        )

    add(
        "domain",
        True,
        f"{DOMAIN} only (fixed; topic must be a subtopic)",
    )
    add(
        "OPENROUTER_API_KEY",
        bool(settings.openrouter_api_key),
        "set" if settings.openrouter_api_key else "not set - live runs will fail closed",
    )
    add(
        "model (strong)",
        True,
        f"{settings.model_strong} | {settings.model_fast}",
    )
    providers = []
    for name in settings.search_providers:
        if name == "duckduckgo" or getattr(settings, f"{name}_api_key", None):
            providers.append(name)
    add(
        "search providers",
        bool(providers),
        ", ".join(providers) if providers else "none configured - web_search will return no results",
    )
    add(
        "network access",
        settings.allow_network,
        "enabled" if settings.allow_network else "disabled (offline mode)",
    )
    add("cache directory", True, str(settings.cache_dir))
    add("output directory", True, str(settings.output_dir))
    add("state directory", True, str(settings.state_dir))
    try:
        settings.validate()
    except ConfigError as exc:
        add("configuration", False, str(exc))
    else:
        add("configuration", True, "valid")
    for warning in settings.warnings():
        add("warning", False, warning)

    if offline:
        add("network checks", True, "skipped (offline mode)")
        console.print(table)
        return

    if settings.openrouter_api_key:
        try:
            import httpx

            response = httpx.get(
                "https://openrouter.ai/api/v1/models", timeout=10, headers={"User-Agent": "mars"}
            )
            add(
                "openrouter reachable",
                response.status_code == 200,
                f"HTTP {response.status_code}",
            )
        except Exception as exc:  # pragma: no cover - network dependent
            add("openrouter reachable", False, str(exc))

    try:
        import httpx

        response = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": "agentic AI"},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        add("duckduckgo reachable", response.status_code == 200, f"HTTP {response.status_code}")
    except Exception as exc:  # pragma: no cover - network dependent
        add("duckduckgo reachable", False, str(exc))

    console.print(table)


@app.command()
def domain() -> None:
    """Show the fixed research domain and its canonical track taxonomy."""
    table = Table(title=f"Mars domain: {DOMAIN}", show_header=True, header_style="bold")
    table.add_column("Track")
    table.add_column("Title")
    table.add_column("Objective")
    for track in DOMAIN_TRACKS:
        table.add_row(track.key, track.title, track.objective)
    console.print(
        Panel.fit(
            f"Mars researches only [bold]{DOMAIN}[/].\n"
            "Off-domain topics are rejected before any model call.\n\n"
            "Example subtopics:\n- " + "\n- ".join(SUBTOPIC_EXAMPLES),
            title="domain lock",
        )
    )
    console.print(table)


@app.command()
def models() -> None:
    """Show the pinned model registry and validate the configured models."""
    from .model_registry import MODEL_REGISTRY, ModelRegistryError, model_pin, validate_model

    settings = build_settings()
    table = Table(title="Mars model registry", show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Released")
    table.add_column("Replacement")
    table.add_column("Context")
    for pin in sorted(MODEL_REGISTRY.values(), key=lambda item: item.model_id):
        table.add_row(
            pin.model_id,
            pin.status,
            pin.released or "-",
            pin.replacement or "-",
            f"{pin.context_window:,}" if pin.context_window else "-",
        )
    console.print(table)

    configured = Table(title="Configured pins", show_header=True, header_style="bold")
    configured.add_column("Role")
    configured.add_column("Model")
    configured.add_column("Status")
    configured.add_column("Notes")
    failed = False
    for role, model in (
        ("strong", settings.model_strong),
        ("fast", settings.model_fast),
        ("planner", settings.model_planner),
        *((f"fallback[{index}]", item) for index, item in enumerate(settings.model_fallbacks)),
    ):
        pin = model_pin(model)
        status = pin.status if pin is not None else "unpinned"
        try:
            warnings = validate_model(model, role=role)
        except ModelRegistryError as exc:
            failed = True
            configured.add_row(role, model, "[red]retired[/]", str(exc))
            continue
        configured.add_row(role, model, status, "; ".join(warnings) or "ok")
    console.print(configured)
    if failed:
        raise typer.Exit(code=1)


@app.command()
def prompts() -> None:
    """Show the active prompt bundle, its version, channel and checksum."""
    from .agents.prompt_registry import PromptRegistryError, load_prompt_bundle

    settings = build_settings()
    try:
        bundle = load_prompt_bundle(
            settings.prompt_overrides_path,
            version=settings.prompt_version,
            channel=settings.prompt_channel,
        )
    except PromptRegistryError as exc:
        console.print(f"[red]Prompt configuration error:[/] {exc}")
        raise typer.Exit(code=2) from None

    console.print(
        Panel.fit(
            f"Version: [bold]{bundle.version}[/]\n"
            f"Channel: [bold]{bundle.channel}[/]\n"
            f"Released: {bundle.released_at}\n"
            f"Source: {bundle.source}\n"
            f"Checksum: {bundle.checksum()}\n"
            f"Agent suffixes: {', '.join(sorted(bundle.agent_suffixes)) or 'none'}",
            title="active prompt bundle",
        )
    )
    if bundle.changelog:
        table = Table(title="Changelog", show_header=True, header_style="bold")
        table.add_column("Entry")
        for entry in bundle.changelog:
            table.add_row(entry)
        console.print(table)


def _load_reports(paths: list[Path]) -> list[dict]:
    """Load saved report JSON files, ignoring plans, sources and run manifests."""
    loaded: list[dict] = []
    for path in paths:
        candidates = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.name.endswith(("-run.json", "-plan.json", "-sources.json")):
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                console.print(f"[yellow]warning:[/] skipping {candidate}: {exc}")
                continue
            if isinstance(payload, dict) and "sections" in payload and "audit" in payload:
                loaded.append(payload)
    return loaded


@app.command("eval")
def eval_command(
    dataset: Path = typer.Option(
        Path("evals/agentic_ai_v1.json"),
        "--dataset",
        "-d",
        help="Versioned evaluation dataset to run.",
    ),
    reports: list[Path] = typer.Option(
        [],
        "--reports",
        "-r",
        help="Saved report JSON file or directory to evaluate research cases against.",
    ),
    baseline: Path | None = typer.Option(
        None,
        "--baseline",
        help="Baseline report JSON; fails when grounding/entailment regress or cost spikes.",
    ),
    max_drop: float = typer.Option(
        0.05, "--max-drop", help="Allowed drop in grounding/entailment ratio."
    ),
    max_cost_increase: float = typer.Option(
        0.25, "--max-cost-increase", help="Allowed fractional increase in estimated cost."
    ),
) -> None:
    """Run the fixed regression set and compare saved reports to a baseline."""
    from .evaluation_set import (
        EvaluationSetError,
        compare_metrics,
        evaluate_dataset,
        load_evaluation_set,
    )

    try:
        evaluation = load_evaluation_set(dataset)
    except EvaluationSetError as exc:
        console.print(f"[red]Evaluation set error:[/] {exc}")
        raise typer.Exit(code=2) from None

    loaded = _load_reports(list(reports))
    summary = evaluate_dataset(evaluation, loaded)

    table = Table(
        title=f"Evaluation {evaluation.name} v{evaluation.version}",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Case")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Detail")
    for result in summary.results:
        colour = {"pass": "green", "fail": "red", "skipped": "yellow"}[result.status]
        table.add_row(
            result.case_id,
            result.kind,
            f"[{colour}]{result.status}[/]",
            result.message,
        )
    console.print(table)
    console.print(
        f"passed={summary.passed} failed={summary.failed} skipped={summary.skipped}"
    )

    regressions: list[str] = []
    if baseline is not None:
        try:
            baseline_payload = json.loads(baseline.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            console.print(f"[red]Baseline error:[/] {exc}")
            raise typer.Exit(code=2) from None
        baseline_topic = str(baseline_payload.get("topic") or "")
        current = next(
            (report for report in loaded if str(report.get("topic") or "") == baseline_topic),
            None,
        )
        if current is None:
            console.print(
                f"[yellow]warning:[/] no current report for baseline topic {baseline_topic!r}; "
                "baseline comparison skipped"
            )
        else:
            regressions = compare_metrics(
                baseline_payload,
                current,
                max_drop=max_drop,
                max_cost_increase=max_cost_increase,
            )
            if regressions:
                for message in regressions:
                    console.print(f"[red]regression:[/] {message}")
            else:
                console.print("[green]baseline comparison: no regressions[/]")

    if not summary.ok or regressions:
        raise typer.Exit(code=1)


@app.command()
def approve(
    run_id: str = typer.Option(..., "--run-id", help="Run id from the approval request."),
    checkpoint: str = typer.Option(..., "--checkpoint", help="Checkpoint name: plan, draft, revision."),
    approver: str = typer.Option("human", "--approver", help="Who is approving."),
    approval_dir: Path = typer.Option(Path(".mars_approvals"), "--approval-dir", help="Where approval files are stored."),
    secret: str | None = typer.Option(
        None,
        "--secret",
        envvar="MARS_APPROVAL_SECRET",
        help="HMAC secret shared with the running pipeline.",
    ),
) -> None:
    """Approve a pending human-in-the-loop checkpoint."""
    from datetime import UTC, datetime

    from .approval import sign_payload

    if not secret:
        console.print(
            "[red]MARS_APPROVAL_SECRET (or --secret) is required to sign approvals.[/]"
        )
        raise typer.Exit(code=1)

    approval_dir.mkdir(parents=True, exist_ok=True)
    path = approval_dir / f"{run_id}-{checkpoint}.approved.json"
    payload = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "approver": approver,
        "approved_at": datetime.now(UTC).isoformat(),
    }
    payload["signature"] = sign_payload(secret, payload)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[green]Approved[/] {run_id} / {checkpoint} -> {path}")


@app.command()
def reject(
    run_id: str = typer.Option(..., "--run-id", help="Run id from the approval request."),
    checkpoint: str = typer.Option(..., "--checkpoint", help="Checkpoint name: plan, draft, revision."),
    reason: str = typer.Option(..., "--reason", help="Reason for rejection."),
    approval_dir: Path = typer.Option(Path(".mars_approvals"), "--approval-dir", help="Where approval files are stored."),
    secret: str | None = typer.Option(
        None,
        "--secret",
        envvar="MARS_APPROVAL_SECRET",
        help="HMAC secret shared with the running pipeline.",
    ),
) -> None:
    """Reject a pending human-in-the-loop checkpoint (stops the run)."""
    from datetime import UTC, datetime

    from .approval import sign_payload

    if not secret:
        console.print(
            "[red]MARS_APPROVAL_SECRET (or --secret) is required to sign rejections.[/]"
        )
        raise typer.Exit(code=1)

    approval_dir.mkdir(parents=True, exist_ok=True)
    path = approval_dir / f"{run_id}-{checkpoint}.rejected.json"
    payload = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "reason": reason,
        "rejected_at": datetime.now(UTC).isoformat(),
    }
    payload["signature"] = sign_payload(secret, payload)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[red]Rejected[/] {run_id} / {checkpoint}: {reason} -> {path}")


@app.command()
def approvals(
    approval_dir: Path = typer.Option(Path(".mars_approvals"), "--approval-dir", help="Where approval files are stored."),
) -> None:
    """List pending approval requests."""
    if not approval_dir.exists():
        console.print("[dim]No approvals directory found.[/]")
        return
    requests = sorted(approval_dir.glob("*.request.json"))
    if not requests:
        console.print("[dim]No pending approvals.[/]")
        return
    table = Table(title="Pending approvals", show_header=True, header_style="bold")
    table.add_column("Run ID")
    table.add_column("Checkpoint")
    table.add_column("Topic")
    table.add_column("Requested")
    for req_path in requests:
        try:
            data = json.loads(req_path.read_text(encoding="utf-8"))
            run_id = data.get("run_id", "")
            checkpoint = data.get("checkpoint", "")
            topic = data.get("topic", "")
            requested = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data.get("requested_at", 0)))
            # skip if already approved/rejected
            if (approval_dir / f"{run_id}-{checkpoint}.approved.json").exists():
                continue
            if (approval_dir / f"{run_id}-{checkpoint}.rejected.json").exists():
                continue
            table.add_row(run_id, checkpoint, topic, requested)
        except Exception:
            continue
    console.print(table)


@app.command()
def show(
    path: Path = typer.Argument(..., help="Path to a report JSON produced by a run."),
) -> None:
    """Pretty-print a stored run report."""
    data = json.loads(path.read_text(encoding="utf-8"))
    report = data.get("report", data)
    console.print_json(json.dumps(report.get("metrics", {}), indent=2))


@app.command()
def replay(
    run_id: str = typer.Option(..., "--run-id", help="Run id whose state journal should be resumed."),
    output_dir: Path = typer.Option(
        Path("research_output"), "--output-dir", "-o", help="Where the resumed run writes artefacts."
    ),
    force: bool = typer.Option(
        False, "--force", help="Replay even when the previous attempt completed successfully."
    ),
) -> None:
    """Replay an interrupted or failed run from its durable state journal."""
    from .reliability import DeadLetterQueue, RunStateStore

    overrides: dict[str, object] = {"run_id": run_id, "output_dir": output_dir}
    settings = _validated(build_settings(**overrides))  # type: ignore[arg-type]

    state = RunStateStore.load(settings.state_dir, run_id)
    if state is None:
        console.print(f"[red]No state journal for run_id {run_id!r} in {settings.state_dir}[/]")
        raise typer.Exit(code=1)
    if state.status == "completed" and not force:
        console.print(
            f"[yellow]Run {run_id!r} already completed; pass --force to replay anyway.[/]"
        )
        raise typer.Exit(code=1)
    if force and state.status == "completed":
        # Reopen the journal so the pipeline's resume guard permits a replay.
        state.finish("running", summary={"reopened_for_replay": True})

    queue = DeadLetterQueue(settings.dead_letter_dir, run_id)
    pending = queue.count()
    if pending:
        console.print(f"Replaying {pending} dead-lettered item(s) from {queue.path}")

    from .pipeline import ResearchPipeline

    pipeline = ResearchPipeline(
        settings, reporter=lambda message: console.print(f"[dim]{message}[/]")
    )
    previous_handler = _install_sigterm(pipeline)
    try:
        result = pipeline.run(resume=True)
    except RuntimeError as exc:
        console.print(f"[red]Replay failed:[/] {exc}")
        raise typer.Exit(code=1) from None
    finally:
        import signal

        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
        pipeline.context.close()
    if pending:
        queue.mark_replayed(replay_run_id=pipeline.run_id)
    console.print(f"[green]Replayed[/] {run_id} -> {result.output_paths.get('report_md')}")


@app.command()
def prune(
    days: int | None = typer.Option(None, "--days", help="Override MARS_RETENTION_DAYS."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would be removed."),
) -> None:
    """Apply the retention window to reports, traces, dead letters and caches."""
    from .retention import prune_all

    settings = build_settings()
    report = prune_all(settings, days=days, dry_run=dry_run)
    payload = report.as_dict()
    verb = "would remove" if dry_run else "removed"
    console.print(
        Panel.fit(
            f"{verb} [bold]{payload['removed']}[/] file(s), "
            f"freed {payload['bytes_freed']} byte(s)\n"
            f"kept {payload['kept']} file(s) within {days or settings.retention_days} day(s)",
            title="retention",
            border_style="green" if not dry_run else "yellow",
        )
    )


@app.command()
def health() -> None:
    """Print the liveness/readiness report the health server would serve."""
    from .health import health_snapshot

    report = health_snapshot()
    table = Table(title=f"Mars health: {report.status}", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in report.checks:
        table.add_row(
            check.name,
            "[green]ok[/]" if check.ok else ("[red]fail[/]" if check.critical else "[yellow]warn[/]"),
            check.detail,
        )
    console.print(table)
    if not report.ready:
        raise typer.Exit(code=1)


@app.command()
def metrics() -> None:
    """Print Prometheus text-exposition metrics for the current host."""
    from .health import metrics_text

    console.print(metrics_text(), markup=False, highlight=False)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    port: int = typer.Option(8080, "--port", help="Bind port."),
) -> None:
    """Serve /healthz, /readyz and /metrics for container probes."""
    from .health import run_server

    settings = build_settings()
    console.print(f"[dim]Mars health server on http://{host}:{port}[/]")
    try:
        run_server(host=host, port=port, settings=settings)
    except OSError as exc:
        console.print(f"[red]Could not bind {host}:{port}:[/] {exc}")
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        return


@app.command()
def verify(
    report: Path = typer.Argument(..., help="Path to a report JSON produced by a run."),
    archive_dir: Path | None = typer.Option(
        None, "--archive-dir", help="Evidence archive root (defaults to the configured data dir)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the full verification result as JSON."),
) -> None:
    """Re-verify a persisted report against its immutable evidence snapshots."""
    from .verification import verification_summary, verify_report_file

    settings = build_settings()
    root = archive_dir or (settings.evidence_dir if settings.archive_enabled else None)
    try:
        result = verify_report_file(report, root)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Could not verify {report}:[/] {exc}")
        raise typer.Exit(code=1) from None
    if as_json:
        console.print_json(json.dumps(result, indent=2))
    else:
        console.print(verification_summary(result))
        audit = result.get("entailment", {})
        table = Table(title="Entailment re-audit", show_header=True, header_style="bold")
        table.add_column("Status")
        table.add_column("Count", justify="right")
        for key in ("supported", "partial", "unsupported", "unverified", "total"):
            table.add_row(key, str(audit.get(key, 0)))
        console.print(table)
        if result.get("missing_or_corrupt_snapshots"):
            console.print(
                "[red]Missing or corrupt snapshots:[/] "
                + ", ".join(result["missing_or_corrupt_snapshots"])
            )
    if not result.get("verified"):
        raise typer.Exit(code=1)


@app.command("delete-run")
def delete_run_command(
    run_id: str = typer.Option(..., "--run-id", help="Run id to delete."),
    output_dir: Path = typer.Option(
        Path("research_output"), "--output-dir", "-o", help="Where the run wrote artefacts."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="List what would be deleted."),
) -> None:
    """Delete every artefact for one run (state, DLQ, approvals, traces, report)."""
    from .data_governance import delete_run

    settings = build_settings(output_dir=output_dir)
    report = delete_run(settings, run_id, dry_run=dry_run)
    for path in report.removed:
        console.print(("would remove " if dry_run else "removed ") + path)
    for error in report.errors:
        console.print(f"[red]{error}[/]")
    console.print(
        f"{len(report.removed)} path(s) {'matched' if dry_run else 'deleted'}"
        + (f", {len(report.errors)} error(s)" if report.errors else "")
    )
    if report.errors:
        raise typer.Exit(code=1)


@app.command()
def disclosure() -> None:
    """Show exactly what data leaves this host and where it goes."""
    from .data_governance import disclosure as _disclosure

    console.print_json(json.dumps(_disclosure(build_settings()), indent=2))


def main() -> None:
    """Console script entry point."""
    from .observability import configure_logging

    configure_logging()
    try:
        app()
    except SystemExit as exc:  # pragma: no cover - typer handles this
        raise exc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
