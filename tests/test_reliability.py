"""Focused tests for fleet-wide failure containment."""

from __future__ import annotations

import json

import pytest

from mars.llm import OpenRouterLLM
from mars.reliability import (
    CallBudget,
    CallBudgetExceeded,
    CircuitOpenError,
    DeadLetterQueue,
    RunStateStore,
    RunWatchdog,
    SharedCircuitBreaker,
    WatchdogExpired,
)


def test_shared_breaker_opens_for_every_agent_after_threshold() -> None:
    breaker = SharedCircuitBreaker(failure_threshold=2, recovery_seconds=60)
    breaker.before_call()
    breaker.record_failure()
    breaker.before_call()
    breaker.record_failure()

    with pytest.raises(CircuitOpenError):
        breaker.before_call()
    assert breaker.snapshot()["state"] == "open"


def test_budget_hard_stops_additional_provider_calls() -> None:
    budget = CallBudget(max_calls=2)
    budget.reserve()
    budget.reserve()
    with pytest.raises(CallBudgetExceeded):
        budget.reserve()


def test_llm_retries_are_bounded_and_use_shared_budget(monkeypatch) -> None:
    llm = OpenRouterLLM(
        model="test/model",
        cache_enabled=False,
        max_retries=2,
        retry_base_seconds=0,
        retry_max_seconds=0,
        breaker=SharedCircuitBreaker(failure_threshold=10),
        call_budget=CallBudget(max_calls=3),
    )
    attempts = 0

    def transient_failure():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("connection temporarily unavailable")

    monkeypatch.setattr("mars.llm.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="temporarily"):
        llm._invoke_with_resilience(transient_failure)

    assert attempts == 3
    assert llm.call_budget.snapshot()["calls_reserved"] == 3


def test_dlq_records_failed_work_durably(tmp_path) -> None:
    queue = DeadLetterQueue(tmp_path, "run-test")
    path = queue.enqueue("research_track", RuntimeError("provider unavailable"), {"track": "economy"})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "created_at": rows[0]["created_at"],
            "error": "provider unavailable",
            "error_type": "RuntimeError",
            "payload": {"track": "economy"},
            "run_id": "run-test",
            "stage": "research_track",
            "schema_version": 1,
        }
    ]


def test_state_store_writes_atomic_stage_checkpoints(tmp_path) -> None:
    state = RunStateStore(tmp_path, "run-test")
    state.transition("plan", "completed", summary={"tracks": ["economy"]})
    state.finish("completed", summary={"grounding": 0.9})

    persisted = json.loads(state.path.read_text(encoding="utf-8"))
    assert persisted["status"] == "completed"
    assert persisted["stages"]["plan"]["summary"]["tracks"] == ["economy"]
    assert not state.path.with_suffix(".tmp").exists()


def test_watchdog_stops_work_after_a_deadline(monkeypatch) -> None:
    clock = iter((100.0, 103.0))
    monkeypatch.setattr("mars.reliability.time.monotonic", lambda: next(clock))
    watchdog = RunWatchdog(max_run_seconds=2, max_track_seconds=2)

    with pytest.raises(WatchdogExpired, match="run deadline"):
        watchdog.check_run()
