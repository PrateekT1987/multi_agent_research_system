"""Focused tests for local tracing, cost attribution, and evaluation."""

from __future__ import annotations

import json

import pytest

from mars.metrics import UsageTracker
from mars.observability import RunObserver
from mars.pipeline import ResearchPipeline
from mars.schemas import EvaluationResult, ResearchReport


def _read_events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_observer_writes_structured_jsonl_spans(tmp_path) -> None:
    observer = RunObserver(tmp_path / "traces", "run-abc")

    observer.emit("tool.call", tool="web_search", elapsed_ms=12)
    with pytest.raises(RuntimeError, match="boom"):
        with observer.span("research.track", track="history"):
            raise RuntimeError("boom")

    rows = _read_events(observer.path)
    assert [row["name"] for row in rows] == [
        "tool.call",
        "research.track.started",
        "research.track.finished",
    ]
    assert all(row["trace_id"] == "run-abc" for row in rows)
    assert all(row["span_id"] for row in rows)
    assert rows[0]["attributes"] == {"tool": "web_search", "elapsed_ms": 12}
    assert rows[2]["status"] == "error"
    assert rows[2]["attributes"]["error_type"] == "RuntimeError"
    assert rows[2]["attributes"]["elapsed_ms"] >= 0


def test_observer_never_breaks_a_run_when_telemetry_fails(tmp_path) -> None:
    # A file where the trace directory should be makes every write fail.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    observer = RunObserver(blocked, "run-abc")

    observer.emit("tool.call", tool="web_search")

    assert blocked.is_file()


def test_usage_tracker_attributes_cost_to_roles() -> None:
    tracker = UsageTracker()
    tracker.record(
        model="google/gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=500,
        role="researcher",
    )
    tracker.record(
        model="anthropic/claude-sonnet-4.5",
        prompt_tokens=200,
        completion_tokens=100,
        role="editor",
    )

    snapshot = tracker.snapshot()
    assert snapshot["by_role"]["researcher"]["calls"] == 1
    assert snapshot["by_role"]["researcher"]["prompt_tokens"] == 1000
    assert snapshot["by_role"]["editor"]["completion_tokens"] == 100
    assert snapshot["by_role"]["editor"]["estimated_cost_usd"] > 0
    assert snapshot["estimated_cost_usd"] == pytest.approx(
        sum(role["estimated_cost_usd"] for role in snapshot["by_role"].values())
    )


def test_usage_merge_preserves_role_breakdown() -> None:
    local = UsageTracker()
    local.record(
        model="google/gemini-2.5-flash",
        prompt_tokens=10,
        completion_tokens=5,
        role="planner",
    )
    worker = UsageTracker()
    worker.record(
        model="google/gemini-2.5-flash",
        prompt_tokens=20,
        completion_tokens=10,
        role="planner",
    )

    local.merge(worker)

    assert local.by_role["planner"]["calls"] == 2
    assert local.by_role["planner"]["prompt_tokens"] == 30
    assert local.by_role["planner"]["completion_tokens"] == 15


def test_evaluation_is_opt_in(settings, tmp_path) -> None:
    settings = settings.with_overrides(cache_dir=tmp_path / "cache")
    pipeline = ResearchPipeline(settings)
    try:
        report = ResearchReport(topic="agentic AI", title="T", draft_markdown="body")
        assert pipeline._evaluate(report) is None
        assert "continuous_evaluation" not in pipeline.context.run_state.snapshot()["stages"]
    finally:
        pipeline.context.close()


def test_evaluation_records_the_judge_result(settings, tmp_path) -> None:
    settings = settings.with_overrides(
        enable_continuous_evaluation=True,
        cache_dir=tmp_path / "cache",
        observability_dir=tmp_path / "traces",
    )
    pipeline = ResearchPipeline(settings)

    class _StubJudge:
        def call(self, prompt: str, response_model=None):
            assert "independent research-quality judge" in prompt
            return response_model(score=0.8, verdict="pass", rationale="well grounded")

    pipeline._llm = lambda role: _StubJudge()
    try:
        report = ResearchReport(topic="agentic AI", title="T", draft_markdown="body")
        result = pipeline._evaluate(report)

        assert isinstance(result, EvaluationResult)
        assert result.score == 0.8
        assert result.verdict == "pass"

        stage = pipeline.context.run_state.snapshot()["stages"]["continuous_evaluation"]
        assert stage["status"] == "completed"
        assert stage["summary"]["verdict"] == "pass"

        events = _read_events(pipeline.context.observer.path)
        judged = [event for event in events if event["name"] == "evaluation.completed"]
        assert judged and judged[-1]["attributes"]["score"] == 0.8
    finally:
        pipeline.context.close()
