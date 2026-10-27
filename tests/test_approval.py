"""Tests for the human-in-the-loop approval gate."""

from __future__ import annotations

import json
import threading
import time

import pytest

from mars.approval import ApprovalGate, sign_payload

SECRET = "test-approval-secret"


def _write_decision(path, payload: dict, secret: str = SECRET) -> None:
    """Write a signed approval/rejection file, as the CLI would."""
    payload = dict(payload)
    payload["signature"] = sign_payload(secret, payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_gate_is_a_no_op_when_approval_is_not_required(settings, tmp_path) -> None:
    gate = ApprovalGate(settings.with_overrides(approval_dir=tmp_path / "approvals"))

    assert gate.request_approval("plan", "run-1", "agentic AI", {"tracks": []}) is True


def test_gate_is_a_no_op_for_checkpoints_that_are_not_configured(settings, tmp_path) -> None:
    gated = settings.with_overrides(
        require_human_approval=True,
        approval_checkpoints=("plan",),
        approval_secret=SECRET,
        approval_dir=tmp_path / "approvals",
    )
    gate = ApprovalGate(gated)

    assert gate.request_approval("draft", "run-1", "agentic AI", {}) is True


def test_gate_requires_a_secret_when_approval_is_enabled(settings, tmp_path) -> None:
    gated = settings.with_overrides(
        require_human_approval=True,
        approval_checkpoints=("plan",),
        approval_dir=tmp_path / "approvals",
    )
    gate = ApprovalGate(gated)

    with pytest.raises(RuntimeError, match="MARS_APPROVAL_SECRET"):
        gate.request_approval("plan", "run-1", "agentic AI", {})


def test_gate_writes_a_request_and_waits_for_approval(settings, tmp_path) -> None:
    gated = settings.with_overrides(
        require_human_approval=True,
        approval_checkpoints=("plan",),
        approval_secret=SECRET,
        approval_dir=tmp_path / "approvals",
    )
    gate = ApprovalGate(gated)

    def approve_soon() -> None:
        time.sleep(0.3)
        _write_decision(
            tmp_path / "approvals" / "run-1-plan.approved.json", {"approver": "test"}
        )

    thread = threading.Thread(target=approve_soon, daemon=True)
    thread.start()

    assert gate.request_approval("plan", "run-1", "agentic AI", {"tracks": ["history"]}) is True
    thread.join(timeout=5)

    request = json.loads((tmp_path / "approvals" / "run-1-plan.request.json").read_text())
    assert request["checkpoint"] == "plan"
    assert request["run_id"] == "run-1"
    assert request["status"] == "pending"


def test_gate_rejects_an_unsigned_approval(settings, tmp_path) -> None:
    gated = settings.with_overrides(
        require_human_approval=True,
        approval_checkpoints=("plan",),
        approval_secret=SECRET,
        approval_timeout_seconds=3,
        approval_dir=tmp_path / "approvals",
    )
    gate = ApprovalGate(gated)

    def approve_soon() -> None:
        time.sleep(0.3)
        (tmp_path / "approvals" / "run-1-plan.approved.json").write_text(
            json.dumps({"approver": "forger"}), encoding="utf-8"
        )

    thread = threading.Thread(target=approve_soon, daemon=True)
    thread.start()

    with pytest.raises(RuntimeError, match="signature"):
        gate.request_approval("plan", "run-1", "agentic AI", {})
    thread.join(timeout=5)


def test_gate_raises_when_the_reviewer_rejects(settings, tmp_path) -> None:
    gated = settings.with_overrides(
        require_human_approval=True,
        approval_checkpoints=("plan",),
        approval_secret=SECRET,
        approval_dir=tmp_path / "approvals",
    )
    gate = ApprovalGate(gated)

    def reject_soon() -> None:
        time.sleep(0.3)
        _write_decision(
            tmp_path / "approvals" / "run-1-plan.rejected.json",
            {"reason": "the plan misses the economics track"},
        )

    thread = threading.Thread(target=reject_soon, daemon=True)
    thread.start()

    with pytest.raises(RuntimeError, match="economics track"):
        gate.request_approval("plan", "run-1", "agentic AI", {})


def test_gate_times_out_when_nobody_answers(settings, tmp_path) -> None:
    gated = settings.with_overrides(
        require_human_approval=True,
        approval_checkpoints=("plan",),
        approval_secret=SECRET,
        approval_timeout_seconds=1,
        approval_dir=tmp_path / "approvals",
    )
    gate = ApprovalGate(gated)

    with pytest.raises(RuntimeError, match="timed out"):
        gate.request_approval("plan", "run-1", "agentic AI", {})


def test_gate_auto_approve_skips_the_wait(settings, tmp_path) -> None:
    gated = settings.with_overrides(
        require_human_approval=True,
        approval_checkpoints=("plan",),
        approval_auto_approve=True,
        approval_secret=SECRET,
        approval_dir=tmp_path / "approvals",
    )
    gate = ApprovalGate(gated)

    assert gate.request_approval("plan", "run-1", "agentic AI", {}) is True
    assert not (tmp_path / "approvals" / "run-1-plan.request.json").exists()


def test_pipeline_pauses_at_the_plan_checkpoint(settings, tmp_path) -> None:
    """The offline pipeline stops between the plan and the research."""
    gated = settings.with_overrides(
        require_human_approval=True,
        approval_checkpoints=("plan",),
        approval_secret=SECRET,
        approval_dir=tmp_path / "approvals",
        output_dir=tmp_path / "out",
    )

    from mars.pipeline import ResearchPipeline

    pipeline = ResearchPipeline(gated)
    pipeline.run_id = "run-pipeline-test"

    def approve_soon() -> None:
        time.sleep(0.5)
        _write_decision(
            tmp_path / "approvals" / "run-pipeline-test-plan.approved.json",
            {"approver": "test"},
        )

    thread = threading.Thread(target=approve_soon, daemon=True)
    thread.start()

    result = pipeline.run()
    thread.join(timeout=5)
    pipeline.context.close()

    assert result.report.topic == gated.topic
    assert (tmp_path / "approvals" / "run-pipeline-test-plan.request.json").exists()
