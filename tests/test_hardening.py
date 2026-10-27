"""P1 production-hardening tests: validation, budgets, shared limits, health."""

from __future__ import annotations

import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from mars.config import ConfigError, Settings, build_settings
from mars.health import health_snapshot, metrics_text
from mars.metrics import SpendBudgetExceeded, UsageTracker, estimate_run_cost, pricing_known
from mars.net import HttpClient, Response, get_shared_rate_limiter
from mars.observability import send_alert
from mars.reliability import (
    CircuitOpenError,
    SharedCircuitBreaker,
    get_shared_circuit_breaker,
)
from mars.retention import prune_all


# --------------------------------------------------------------------- config
def test_validate_rejects_missing_live_key() -> None:
    settings = build_settings(llm_provider="openrouter", env_file=None).with_overrides(
        openrouter_api_key=""
    )

    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        settings.validate()


def test_validate_rejects_impossible_limits() -> None:
    settings = build_settings(llm_provider="openrouter", openrouter_api_key="sk-test", env_file=None).with_overrides(
        max_rounds=0,
        max_parallel_tracks=0,
        trace_sample_rate=2.0,
        cache_max_bytes=0,
    )

    with pytest.raises(ConfigError) as excinfo:
        settings.validate()

    message = str(excinfo.value)
    assert "max_rounds" in message
    assert "max_parallel_tracks" in message
    assert "trace_sample_rate" in message
    assert "cache_max_bytes" in message


def test_validate_rejects_unknown_providers_and_bad_urls() -> None:
    settings = build_settings(llm_provider="openrouter", openrouter_api_key="sk-test", env_file=None).with_overrides(
        search_providers=("duckduckgo", "made-up"),
        openrouter_base_url="not-a-url",
    )

    with pytest.raises(ConfigError) as excinfo:
        settings.validate()

    assert "made-up" in str(excinfo.value)
    assert "openrouter_base_url" in str(excinfo.value)


def test_strict_pricing_refuses_an_unpriced_model() -> None:
    settings = build_settings(llm_provider="openrouter", openrouter_api_key="sk-test", env_file=None).with_overrides(
        model_fast="vendor/unknown-model",
        model_strong="vendor/unknown-model",
        model_planner="vendor/unknown-model",
        strict_pricing=True,
    )

    with pytest.raises(ConfigError, match="strict pricing"):
        settings.validate()


def test_warnings_flag_unpriced_models() -> None:
    settings = build_settings(llm_provider="openrouter", openrouter_api_key="sk-test", env_file=None).with_overrides(
        model_fast="vendor/unknown-model",
        model_strong="vendor/unknown-model",
        model_planner="vendor/unknown-model",
    )

    assert any("pricing" in warning for warning in settings.warnings())


def test_state_dir_is_derived_from_the_cache_dir(tmp_path) -> None:
    settings = build_settings(
        llm_provider="openrouter", openrouter_api_key="sk-test",
        cache_dir=tmp_path / "cache", env_file=None
    )

    assert settings.state_dir == tmp_path / "runs"


def test_state_dir_can_be_overridden(tmp_path) -> None:
    settings = build_settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        cache_dir=tmp_path / "cache",
        run_state_dir=tmp_path / "states",
        env_file=None,
    )

    assert settings.state_dir == tmp_path / "states"


def test_secret_can_be_mounted_as_a_file(tmp_path, monkeypatch) -> None:
    secret = tmp_path / "openrouter-key"
    secret.write_text("sk-mounted\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY_FILE", str(secret))

    settings = build_settings(llm_provider="openrouter", env_file=None)

    assert settings.openrouter_api_key == "sk-mounted"
    assert "sk-mounted" not in json.dumps(settings.to_dict())


# --------------------------------------------------------------------- budgets
def test_usage_tracker_enforces_token_ceiling() -> None:
    usage = UsageTracker(max_tokens=10)
    usage.record(model="mock/deterministic", prompt_tokens=6, completion_tokens=4)

    with pytest.raises(SpendBudgetExceeded, match="token budget"):
        usage.check_budget()


def test_usage_tracker_enforces_cost_ceiling() -> None:
    usage = UsageTracker(max_cost_usd=0.001)
    usage.record(
        model="anthropic/claude-sonnet-4.5", prompt_tokens=1000, completion_tokens=1000
    )

    with pytest.raises(SpendBudgetExceeded, match="cost budget"):
        usage.check_budget()


def test_unknown_models_are_marked_unpriced_not_free() -> None:
    usage = UsageTracker()
    usage.record(model="vendor/unknown-model", prompt_tokens=100, completion_tokens=100)

    snapshot = usage.snapshot()
    assert snapshot["cost_known"] is False
    assert snapshot["unpriced_models"] == ["vendor/unknown-model"]


def test_configured_nvidia_model_is_priced_and_known() -> None:
    assert pricing_known("nvidia/nemotron-3-ultra-550b-a55b:free")
    cost, tokens = estimate_run_cost("nvidia/nemotron-3-ultra-550b-a55b:free", "standard")
    assert cost == 0.0
    assert tokens > 0


def test_unpriced_run_cost_is_none() -> None:
    cost, tokens = estimate_run_cost("vendor/unknown-model", "quick")
    assert cost is None
    assert tokens > 0


# ---------------------------------------------------------------- reliability
def test_shared_breaker_is_process_wide_and_opens() -> None:
    first = get_shared_circuit_breaker("unit-test-provider")
    second = get_shared_circuit_breaker("unit-test-provider")
    assert first is second

    for _ in range(first.failure_threshold):
        first.record_failure()

    with pytest.raises(CircuitOpenError):
        first.before_call()
    assert first.snapshot()["state"] == "open"

    first.record_success()
    first.before_call()
    assert first.snapshot()["state"] == "closed"


def test_breaker_half_opens_after_recovery_window() -> None:
    breaker = SharedCircuitBreaker(failure_threshold=1, recovery_seconds=0.01)
    breaker.record_failure()
    time.sleep(0.02)

    breaker.before_call()  # acquires the single probe
    with pytest.raises(CircuitOpenError, match="half-open"):
        breaker.before_call()
    assert breaker.snapshot()["state"] == "half_open"


def test_rate_limiter_is_shared_per_policy() -> None:
    first = get_shared_rate_limiter(17, 0.25)
    second = get_shared_rate_limiter(17, 0.25)
    other = get_shared_rate_limiter(18, 0.25)

    assert first is second
    assert first is not other
    assert first.snapshot()["requests_per_minute"] == 17


# ------------------------------------------------------------------ cache caps
def _response(url: str) -> Response:
    return Response(url=url, status=200, text="body", headers={"content-type": "text/plain"})


def test_http_cache_evicts_oldest_entries_beyond_cap(tmp_path) -> None:
    client = HttpClient(cache_dir=tmp_path, cache_max_entries=2, cache_max_bytes=10_000)
    for index in range(5):
        client._cache_write(f"key-{index}", _response(f"https://example.com/{index}"))
        time.sleep(0.01)

    stats = client.cache_stats()
    assert stats["entries"] <= 2
    assert client._cache_read("key-0") is None
    assert client._cache_read("key-4") is not None


def test_http_cache_ttl_expiry(tmp_path) -> None:
    client = HttpClient(cache_dir=tmp_path, cache_ttl_hours=1)
    client._cache_write("stale", _response("https://example.com/stale"))
    path = client._cache_path("stale")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["fetched_at"] = time.time() - 7200
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert client._cache_read("stale") is None


# ------------------------------------------------------------------- retention
def test_prune_all_removes_only_expired_artefacts(tmp_path) -> None:
    settings = build_settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        observability_dir=tmp_path / "traces",
        dead_letter_dir=tmp_path / "dead",
        approval_dir=tmp_path / "approvals",
        retention_days=7,
        env_file=None,
    )
    old = settings.output_dir / "old-run.md"
    fresh = settings.output_dir / "fresh-run.md"
    old.parent.mkdir(parents=True)
    old.write_text("old", encoding="utf-8")
    fresh.write_text("fresh", encoding="utf-8")
    expired = time.time() - 8 * 86_400
    import os

    os.utime(old, (expired, expired))

    report = prune_all(settings)

    assert not old.exists()
    assert fresh.exists()
    assert report.bytes_freed >= 3


def test_prune_all_dry_run_keeps_files(tmp_path) -> None:
    settings = build_settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        retention_days=0,
        env_file=None,
    )
    target = settings.output_dir / "old.md"
    target.parent.mkdir(parents=True)
    target.write_text("x", encoding="utf-8")
    expired = time.time() - 86_400
    import os

    os.utime(target, (expired, expired))

    report = prune_all(settings, dry_run=True)

    assert target.exists()
    assert report.dry_run is True
    assert str(target) in report.removed


# ---------------------------------------------------------------------- health
def _healthy_settings(tmp_path) -> Settings:
    return build_settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        dead_letter_dir=tmp_path / "dead",
        observability_dir=tmp_path / "traces",
        approval_dir=tmp_path / "approvals",
        env_file=None,
    )


def test_health_snapshot_reports_ready(tmp_path) -> None:
    report = health_snapshot(_healthy_settings(tmp_path))

    assert report.ready is True
    assert report.status in {"ok", "degraded"}
    assert {check.name for check in report.checks} >= {
        "configuration",
        "data_dir_writable",
        "output_dir_writable",
    }


def test_health_snapshot_is_unhealthy_for_invalid_config(tmp_path) -> None:
    settings = _healthy_settings(tmp_path).with_overrides(
        llm_provider="openrouter", openrouter_api_key=""
    )

    report = health_snapshot(settings)

    assert report.ready is False
    assert report.status == "unhealthy"


def test_metrics_text_exposes_prometheus_gauges(tmp_path) -> None:
    text = metrics_text(_healthy_settings(tmp_path))

    assert "mars_up 1" in text
    assert "mars_ready 1" in text
    assert "mars_cache_bytes" in text
    assert "mars_run_states" in text


def test_send_alert_posts_json(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        status_code = 204

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)

    assert send_alert("https://alerts.example/hook", "run_failed", "boom", run_id="r1") is True
    assert calls[0]["json"]["kind"] == "run_failed"
    assert calls[0]["json"]["fields"]["run_id"] == "r1"


def test_send_alert_is_best_effort(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise OSError("no route")

    monkeypatch.setattr("httpx.post", boom)

    assert send_alert("https://alerts.example/hook", "run_failed", "boom") is False
    assert send_alert(None, "run_failed", "boom") is False


def test_health_handler_serves_endpoints(tmp_path) -> None:
    from mars.health import _Handler

    settings = _healthy_settings(tmp_path)
    handler = type("TestHandler", (_Handler,), {"settings": settings})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert httpx.get(f"{base}/healthz", timeout=5).status_code == 200
        assert httpx.get(f"{base}/readyz", timeout=5).status_code == 200
        metrics = httpx.get(f"{base}/metrics", timeout=5)
        assert metrics.status_code == 200
        assert "mars_up" in metrics.text
        assert httpx.get(f"{base}/missing", timeout=5).status_code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_run_state_journal_survives_corruption(tmp_path) -> None:
    from mars.reliability import RunStateStore

    path = Path(tmp_path) / "run-x.state.json"
    path.write_text("{broken", encoding="utf-8")

    store = RunStateStore(tmp_path, "run-x")

    assert store.status == "running"
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "run-x"
