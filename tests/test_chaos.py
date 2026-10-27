"""Chaos/fault-injection tests for the failure paths that matter in production."""

from __future__ import annotations

import json
import threading

import pytest

from mars.config import build_settings
from mars.metrics import SpendBudgetExceeded, UsageTracker
from mars.reliability import (
    CallBudget,
    CallBudgetExceeded,
    CircuitOpenError,
    RunWatchdog,
    SharedCircuitBreaker,
    WatchdogExpired,
)


def test_call_budget_is_thread_safe_and_exhausts_exactly() -> None:
    budget = CallBudget(50)
    successes = 0
    failures = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal successes, failures
        for _ in range(10):
            try:
                budget.reserve()
            except CallBudgetExceeded:
                with lock:
                    failures += 1
            else:
                with lock:
                    successes += 1

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert successes == 50
    assert failures == 50


def test_shared_breaker_sheds_load_across_concurrent_callers() -> None:
    breaker = SharedCircuitBreaker(failure_threshold=1, recovery_seconds=60)
    breaker.record_failure()
    errors: list[Exception] = []

    def call() -> None:
        try:
            breaker.before_call()
        except CircuitOpenError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 20


def test_watchdog_stops_a_cooperating_stage() -> None:
    watchdog = RunWatchdog(max_run_seconds=60, max_track_seconds=60)
    watchdog.request_stop()

    with pytest.raises(WatchdogExpired, match="shutdown"):
        watchdog.check_run()
    assert watchdog.stop_requested is True


def test_watchdog_enforces_track_deadline() -> None:
    watchdog = RunWatchdog(max_run_seconds=60, max_track_seconds=0.01)

    with pytest.raises(WatchdogExpired, match="track"):
        watchdog.check_track(watchdog.started - 1, "history")


def test_usage_budget_failure_does_not_corrupt_counters() -> None:
    usage = UsageTracker(max_tokens=10)
    usage.record(model="mock/deterministic", prompt_tokens=10, completion_tokens=0)

    with pytest.raises(SpendBudgetExceeded):
        usage.check_budget()
    assert usage.snapshot()["total_tokens"] == 10


def test_state_store_rejects_another_runs_journal(tmp_path) -> None:
    from mars.reliability import RunStateStore

    first = RunStateStore(tmp_path, "run-a")
    first.transition("plan", "completed", summary={"tracks": ["history"]})

    # A different run id must not read run-a's state.
    second = RunStateStore(tmp_path, "run-b")
    assert second.stage_status("plan") is None
    assert json.loads((tmp_path / "run-b.state.json").read_text(encoding="utf-8"))["run_id"] == "run-b"


def test_dead_letter_replay_marker(tmp_path) -> None:
    from mars.reliability import DeadLetterQueue

    queue = DeadLetterQueue(tmp_path, "run-x")
    queue.enqueue("track", RuntimeError("boom"), {"track": "history"})
    queue.enqueue("track", RuntimeError("again"), {"track": "future"})

    assert queue.count() == 2
    marker = queue.mark_replayed(replay_run_id="run-x")
    payload = json.loads(marker.read_text(encoding="utf-8"))

    assert payload["items"] == 2
    assert payload["replay_run_id"] == "run-x"
    assert [item["error"] for item in queue.read_items()] == ["boom", "again"]


def test_observer_write_failure_is_counted_not_raised(tmp_path, monkeypatch) -> None:
    from mars.observability import RunObserver

    observer = RunObserver(tmp_path / "traces", "run-observe")

    def boom(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("mars.observability.append_jsonl", boom)
    observer.emit("tool.call", tool="web_search")

    snapshot = observer.snapshot()
    assert snapshot["emitted"] == 1
    assert snapshot["write_errors"] == 1


def test_resume_refuses_a_completed_run(tmp_path) -> None:
    from mars.pipeline import ResearchPipeline

    settings = build_settings(
        topic="agentic AI",
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        cache_enabled=False,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        dead_letter_dir=tmp_path / "dead",
        observability_dir=tmp_path / "traces",
        approval_dir=tmp_path / "approvals",
        run_id="run-completed",
        max_rounds=1,
        env_file=None,
        allow_network=False,
        extra={"fake_llm": True, "test_corpus": True},
    )
    first = ResearchPipeline(settings)
    try:
        first.run()
    finally:
        first.context.close()

    second = ResearchPipeline(settings)
    try:
        with pytest.raises(RuntimeError, match="already completed"):
            second.run(resume=True)
    finally:
        second.context.close()


def test_resume_reuses_completed_tracks(tmp_path) -> None:
    from mars.pipeline import ResearchPipeline

    settings = build_settings(
        topic="agentic AI",
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        cache_enabled=False,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        dead_letter_dir=tmp_path / "dead",
        observability_dir=tmp_path / "traces",
        approval_dir=tmp_path / "approvals",
        run_id="run-resume",
        max_rounds=1,
        env_file=None,
        allow_network=False,
        extra={"fake_llm": True, "test_corpus": True},
    )
    first = ResearchPipeline(settings)
    try:
        original = first.run()
    finally:
        first.context.close()

    state_path = settings.state_dir / "run-resume.state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completed_tracks = {
        name: entry
        for name, entry in state["stages"].items()
        if name.startswith("track:") and entry["status"] == "completed"
    }
    assert completed_tracks
    # Simulate a crash after research but before the report was finished.
    state["status"] = "running"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    second = ResearchPipeline(settings)
    messages: list[str] = []
    second.reporter = messages.append
    try:
        resumed = second.run(resume=True)
    finally:
        second.context.close()

    assert any("Reusing" in message for message in messages)
    assert resumed.report.title == original.report.title
    # The hermetic run is degraded (no entailment evidence), so the artefact is
    # deliberately labelled .DEGRADED.md.
    assert resumed.output_paths["report_md"].endswith(
        ("run-resume.md", "run-resume.DEGRADED.md")
    )
    assert (tmp_path / "out" / "agentic-ai-run-resume-sources.json").exists()
