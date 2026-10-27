"""Tests for per-agent and per-track cost attribution and budgets."""

from __future__ import annotations

import pytest

from mars.metrics import (
    SpendBudgetExceeded,
    UsageTracker,
    current_agent,
    current_track,
    usage_scope,
)
from mars.pipeline import ResearchPipeline, _cost_drivers

FREE = "nvidia/nemotron-3-ultra-550b-a55b:free"
PAID = "anthropic/claude-sonnet-4.5"


def test_record_attributes_calls_to_agent_and_track() -> None:
    tracker = UsageTracker()
    tracker.record(
        model=FREE,
        prompt_tokens=10,
        completion_tokens=5,
        agent="writer",
        track="safety",
    )
    tracker.record(
        model=FREE,
        prompt_tokens=4,
        completion_tokens=1,
        agent="auditor",
        track="report",
    )

    snapshot = tracker.snapshot()
    assert snapshot["by_agent"]["writer"]["calls"] == 1
    assert snapshot["by_agent"]["writer"]["prompt_tokens"] == 10
    assert snapshot["by_agent"]["auditor"]["completion_tokens"] == 1
    assert snapshot["by_track"]["safety"]["calls"] == 1
    assert snapshot["by_track"]["report"]["calls"] == 1


def test_usage_scope_sets_and_restores_attribution() -> None:
    tracker = UsageTracker()
    with usage_scope(agent="researcher", track="foundations"):
        assert current_agent() == "researcher"
        assert current_track() == "foundations"
        tracker.record(model=FREE, prompt_tokens=1, completion_tokens=1)

    assert current_agent() == "unknown"
    assert current_track() == "run"
    assert "researcher" in tracker.snapshot()["by_agent"]
    assert "foundations" in tracker.snapshot()["by_track"]


def test_usage_scope_can_set_one_dimension() -> None:
    tracker = UsageTracker()
    with usage_scope(agent="editor"):
        tracker.record(model=FREE, prompt_tokens=1, completion_tokens=1)
    snapshot = tracker.snapshot()
    assert snapshot["by_agent"]["editor"]["calls"] == 1
    assert snapshot["by_track"]["run"]["calls"] == 1


def test_per_track_token_budget_fails_closed() -> None:
    tracker = UsageTracker(max_tokens_per_track=10)
    tracker.record(model=FREE, prompt_tokens=6, completion_tokens=6, track="safety")
    with usage_scope(track="safety"):
        with pytest.raises(SpendBudgetExceeded, match="track 'safety' token budget"):
            tracker.check_budget()
    with usage_scope(track="foundations"):
        tracker.check_budget()  # a different track is unaffected


def test_per_track_cost_budget_fails_closed() -> None:
    tracker = UsageTracker(max_cost_usd_per_track=0.000001)
    tracker.record(model=PAID, prompt_tokens=10_000, completion_tokens=1_000, track="safety")
    with usage_scope(track="safety"):
        with pytest.raises(SpendBudgetExceeded, match="track 'safety' cost budget"):
            tracker.check_budget()


def test_merge_preserves_agent_and_track_buckets() -> None:
    left = UsageTracker()
    right = UsageTracker()
    left.record(model=FREE, prompt_tokens=1, completion_tokens=1, agent="writer", track="safety")
    right.record(model=FREE, prompt_tokens=2, completion_tokens=2, agent="writer", track="safety")
    right.record(model=FREE, prompt_tokens=3, completion_tokens=3, agent="auditor", track="report")

    left.merge(right)
    snapshot = left.snapshot()
    assert snapshot["by_agent"]["writer"]["calls"] == 2
    assert snapshot["by_agent"]["writer"]["prompt_tokens"] == 3
    assert snapshot["by_agent"]["auditor"]["calls"] == 1
    assert snapshot["by_track"]["safety"]["calls"] == 2
    assert snapshot["by_track"]["report"]["calls"] == 1


def test_cost_drivers_name_the_top_agent_and_track() -> None:
    tracker = UsageTracker()
    tracker.record(model=FREE, prompt_tokens=10, completion_tokens=1, agent="writer", track="safety")
    tracker.record(model=FREE, prompt_tokens=90, completion_tokens=9, agent="auditor", track="report")
    drivers = _cost_drivers(tracker.snapshot())
    assert drivers["top_agent_by_tokens"]["name"] == "auditor"
    assert drivers["top_track_by_tokens"]["name"] == "report"
    assert drivers["top_agent_by_cost"]["name"] in {"writer", "auditor"}


def test_hermetic_pipeline_records_every_agent_and_track(settings) -> None:
    result = ResearchPipeline(settings).run()
    snapshot = result.usage.snapshot()

    assert {"director", "researcher", "analyst", "writer", "editor", "auditor"} <= set(
        snapshot["by_agent"]
    )
    assert "plan" in snapshot["by_track"]
    assert "report" in snapshot["by_track"]
    assert set(settings.tracks) & set(snapshot["by_track"])

    drivers = result.report.metrics["cost_drivers"]
    assert drivers["top_agent_by_cost"]["name"] in snapshot["by_agent"]
    assert drivers["top_track_by_cost"]["name"] in snapshot["by_track"]
