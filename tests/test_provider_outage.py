"""Provider-outage behaviour: rate limits, the shared breaker, and degradation.

The incident these tests encode: a burst of OpenRouter ``429`` responses
opened the fleet-wide breaker within a second, every track failed instantly,
and the editorial call raised out of the pipeline. A run that had already paid
for research ended in a traceback instead of a report.

The contract is now:

* a hard free-tier quota is fatal and explicit, never retried;
* a transient rate limit honours ``Retry-After`` and counts one breaker
  failure per call, not per attempt;
* the breaker can be waited out instead of failing every agent at once;
* a failed editor degrades to a mechanical draft;
* a run with no completed track stops with a clean, actionable error.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

import mars.pipeline as pipeline_module
from mars.llm import OpenRouterLLM
from mars.pipeline import ResearchPipeline
from mars.reliability import (
    CircuitOpenError,
    ProviderUnavailableError,
    SharedCircuitBreaker,
)


class _ProviderError(RuntimeError):
    """Stand-in for an OpenRouter SDK error with a status and headers."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 429,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


def _llm(*, breaker: SharedCircuitBreaker | None = None, max_retries: int = 2) -> OpenRouterLLM:
    return OpenRouterLLM(
        model="test/model",
        cache_enabled=False,
        max_retries=max_retries,
        retry_base_seconds=0,
        retry_max_seconds=0,
        circuit_wait_seconds=0,
        breaker=breaker,
    )


# --------------------------------------------------------------- rate limits
def test_hard_free_tier_quota_is_fatal_and_not_retried(monkeypatch) -> None:
    breaker = SharedCircuitBreaker(failure_threshold=2)
    llm = _llm(breaker=breaker)
    sleeps: list[float] = []
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise _ProviderError(
            "Error code: 429 - Rate limit exceeded: free-models-per-day",
            headers={"Retry-After": "30"},
        )

    monkeypatch.setattr("mars.llm.time.sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="hard quota"):
        llm._invoke_with_resilience(operation)

    assert attempts == 1
    assert sleeps == []
    # A spent quota is not a provider outage, so it must not trip the breaker.
    assert breaker.snapshot()["failures"] == 0


def test_transient_rate_limit_honours_retry_after(monkeypatch) -> None:
    breaker = SharedCircuitBreaker(failure_threshold=10)
    llm = _llm(breaker=breaker)
    sleeps: list[float] = []
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise _ProviderError("upstream temporarily unavailable", status=503, headers={"retry-after": "7"})

    monkeypatch.setattr("mars.llm.time.sleep", sleeps.append)

    with pytest.raises(_ProviderError):
        llm._invoke_with_resilience(operation)

    assert attempts == 3
    assert sleeps == [7.0, 7.0]
    # Three attempts are one failed call, so the breaker sees one failure.
    assert breaker.snapshot()["failures"] == 1


def test_non_retryable_client_error_does_not_touch_the_breaker(monkeypatch) -> None:
    breaker = SharedCircuitBreaker(failure_threshold=1)
    llm = _llm(breaker=breaker)
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise _ProviderError("invalid request", status=400)

    monkeypatch.setattr("mars.llm.time.sleep", lambda _: None)

    with pytest.raises(_ProviderError):
        llm._invoke_with_resilience(operation)

    assert attempts == 1
    assert breaker.snapshot()["state"] == "closed"


# ------------------------------------------------------------------- breaker
def test_breaker_can_be_waited_out_instead_of_failing_every_agent() -> None:
    breaker = SharedCircuitBreaker(failure_threshold=1, recovery_seconds=0.05)
    breaker.record_failure()

    started = time.monotonic()
    breaker.before_call(timeout=1.0)
    elapsed = time.monotonic() - started

    assert elapsed >= 0.04
    # The caller that waited becomes the single half-open probe.
    assert breaker.snapshot()["state"] == "half_open"
    breaker.record_success()
    assert breaker.snapshot()["state"] == "closed"


def test_breaker_wait_times_out_with_a_retry_hint() -> None:
    breaker = SharedCircuitBreaker(failure_threshold=1, recovery_seconds=60)
    breaker.record_failure()

    started = time.monotonic()
    with pytest.raises(CircuitOpenError) as info:
        breaker.before_call(timeout=0.05)

    assert time.monotonic() - started < 1.0
    assert info.value.retry_after is not None
    assert info.value.retry_after > 0


def test_parallel_waiters_are_released_when_the_probe_recovers() -> None:
    breaker = SharedCircuitBreaker(failure_threshold=1, recovery_seconds=0.05)
    breaker.record_failure()
    finished: list[float] = []

    def caller() -> None:
        breaker.before_call(timeout=2.0)
        finished.append(time.monotonic())

    thread = threading.Thread(target=caller)
    thread.start()
    thread.join(timeout=1.5)

    assert not thread.is_alive()
    assert len(finished) == 1


def test_fail_fast_is_still_the_default() -> None:
    breaker = SharedCircuitBreaker(failure_threshold=1, recovery_seconds=60)
    breaker.record_failure()

    with pytest.raises(CircuitOpenError, match="circuit is open"):
        breaker.before_call()


def test_a_non_retryable_probe_releases_the_half_open_slot() -> None:
    """A 400 from the probe must not wedge every other agent on the breaker."""
    breaker = SharedCircuitBreaker(failure_threshold=1, recovery_seconds=0.01)
    breaker.record_failure()
    time.sleep(0.02)

    assert breaker.before_call(timeout=0.5) is True
    breaker.release_probe()

    # The slot is free again, so another caller can probe without waiting.
    assert breaker.before_call(timeout=0.5) is True


# ------------------------------------------------------------------ pipeline
def test_editor_failure_degrades_to_a_mechanical_draft(settings, monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise CircuitOpenError("LLM circuit is open; retry after 29.9s")

    monkeypatch.setattr(pipeline_module, "run_editing", unavailable)

    result = ResearchPipeline(settings).run()

    assert "Editorial note" in result.report.draft_markdown
    assert result.report.metrics["edit_status"] == "degraded"
    assert result.report.audit.degraded is True
    assert any(
        "editing agent" in reason for reason in result.report.audit.degraded_reasons
    )
    # The research that was already paid for is still written to disk.
    assert result.output_paths["report_md"]


def test_a_run_with_no_completed_track_stops_with_an_actionable_error(
    settings, monkeypatch
) -> None:
    def unavailable(*args, **kwargs):
        raise CircuitOpenError("LLM circuit is open; retry after 29.9s")

    monkeypatch.setattr(pipeline_module, "run_research", unavailable)

    with pytest.raises(ProviderUnavailableError, match="every research track failed"):
        ResearchPipeline(settings).run()


def test_a_hard_quota_during_planning_stops_before_research(settings, monkeypatch) -> None:
    researched: list[int] = []

    def quota(*args, **kwargs):
        raise ProviderUnavailableError("OpenRouter rate limit is a hard quota")

    monkeypatch.setattr(pipeline_module, "run_director", quota)
    monkeypatch.setattr(
        pipeline_module, "run_research", lambda *a, **k: researched.append(1)
    )

    with pytest.raises(ProviderUnavailableError, match="hard quota"):
        ResearchPipeline(settings).run()

    assert researched == []
