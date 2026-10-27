"""Tests for the Typer CLI."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from mars.cli import app

runner = CliRunner()


def test_research_hermetic_run_writes_a_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MARS_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("MARS_ALLOW_NETWORK", "0")
    monkeypatch.setenv("MARS_TEST_FAKE_LLM", "1")
    monkeypatch.setenv("MARS_TEST_CORPUS", "1")
    monkeypatch.setenv("MARS_ARCHIVE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("MARS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MARS_DEAD_LETTER_DIR", str(tmp_path / "dead"))
    monkeypatch.setenv("MARS_OBSERVABILITY_DIR", str(tmp_path / "traces"))
    result = runner.invoke(
        app,
        ["research", "--output-dir", str(tmp_path), "--rounds", "1", "--quiet"],
    )

    # The hermetic provider produces no entailment evidence, so the run is
    # degraded by design: the artefact is named .DEGRADED.md and the CLI exits
    # 2 so a caller cannot mistake it for a verified report.
    assert result.exit_code == 2, result.output
    reports = list(tmp_path.glob("*.md"))
    assert reports, "no markdown report was written"
    report_path = reports[0]
    assert report_path.name.endswith(".DEGRADED.md")
    report_text = report_path.read_text(encoding="utf-8")
    assert "## References" in report_text
    assert "DEGRADED - not a verified research report" in report_text

    run_manifests = list(tmp_path.glob("*-run.json"))
    assert run_manifests
    manifest = json.loads(run_manifests[0].read_text(encoding="utf-8"))
    assert manifest["settings"]["llm_provider"] == "openrouter"
    assert manifest["settings"]["allow_network"] is False
    assert manifest["metrics"]["sources"] > 0
    assert manifest["schema_version"] == 2
    assert manifest["prompt_version"]
    assert manifest["prompt"]["version"] == manifest["prompt_version"]
    assert manifest["prompt"]["checksum"]
    assert manifest["contracts"]["version"]
    assert set(manifest["contracts"]["agents"]) >= {"director", "researcher", "auditor"}
    assert manifest["model_snapshot"]["strong"]
    assert manifest["model_snapshot"]["registry"]["pins"]["strong"]["status"] == "active"
    assert manifest["metrics"]["by_agent"]
    assert manifest["metrics"]["by_track"]
    assert manifest["metrics"]["cost_drivers"]
    # The hermetic run must never persist secrets.
    assert manifest["http"]["requests"] == 0
    assert manifest["settings"]["openrouter_api_key"] == "***redacted***"
    assert "sk-test" not in json.dumps(manifest)


def test_doctor_runs_without_credentials() -> None:
    result = runner.invoke(app, ["doctor", "--offline"])

    assert result.exit_code == 0, result.output
    assert "OPENROUTER_API_KEY" in result.output
    assert "skipped (offline mode)" in result.output


def test_show_prints_metrics(tmp_path) -> None:
    payload = tmp_path / "run.json"
    payload.write_text(json.dumps({"metrics": {"sources": 7}}), encoding="utf-8")

    result = runner.invoke(app, ["show", str(payload)])

    assert result.exit_code == 0, result.output
    assert "sources" in result.output


def test_research_reports_a_missing_api_key(tmp_path, monkeypatch) -> None:
    # Set to an empty string rather than deleting: build_settings() loads .env,
    # which would otherwise re-supply the real key and start a live run.
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("MARS_LLM_PROVIDER", "openrouter")

    result = runner.invoke(
        app,
        ["research", "--output-dir", str(tmp_path), "--rounds", "1", "--quiet"],
    )

    # Should fail with a clear message about missing API key
    assert result.exit_code == 2
    assert "OPENROUTER_API_KEY" in result.output
    assert "required" in result.output


def test_approve_writes_the_approval_file(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "approve",
            "--run-id",
            "run-1",
            "--checkpoint",
            "plan",
            "--approver",
            "alice",
            "--secret",
            "test-secret",
            "--approval-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "run-1-plan.approved.json").read_text(encoding="utf-8"))
    assert payload["checkpoint"] == "plan"
    assert payload["approver"] == "alice"
    assert payload["signature"]


def test_approve_requires_a_secret(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "approve",
            "--run-id",
            "run-1",
            "--checkpoint",
            "plan",
            "--approval-dir",
            str(tmp_path),
        ],
        env={"MARS_APPROVAL_SECRET": ""},
    )

    assert result.exit_code == 1
    assert "MARS_APPROVAL_SECRET" in result.output
    assert not (tmp_path / "run-1-plan.approved.json").exists()


def test_reject_writes_the_rejection_file(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "reject",
            "--run-id",
            "run-1",
            "--checkpoint",
            "draft",
            "--reason",
            "the draft is not grounded",
            "--secret",
            "test-secret",
            "--approval-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "run-1-draft.rejected.json").read_text(encoding="utf-8"))
    assert payload["reason"] == "the draft is not grounded"
    assert payload["signature"]


def test_approvals_lists_pending_requests(tmp_path) -> None:
    (tmp_path / "run-1-plan.request.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "checkpoint": "plan",
                "topic": "agentic AI",
                "requested_at": 1_700_000_000,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["approvals", "--approval-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "run-1" in result.output
    assert "plan" in result.output


def test_approvals_hides_resolved_requests(tmp_path) -> None:
    (tmp_path / "run-1-plan.request.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "checkpoint": "plan",
                "topic": "agentic AI",
                "requested_at": 1_700_000_000,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run-1-plan.approved.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["approvals", "--approval-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "run-1" not in result.output


def test_verify_reports_a_verified_run(tmp_path) -> None:
    from mars.archive import EvidenceArchive
    from mars.schemas import Source

    claim = "Agents use tools to accomplish goals beyond a single completion."
    archive = EvidenceArchive(tmp_path / "archive")
    record = archive.store(claim, url="https://example.com/agents", title="Agents")
    assert record is not None
    source = Source(
        id="S1",
        url="https://example.com/agents",
        title="Agents",
        snippet=claim,
        content_hash=record.content_hash,
        archive_path=record.path,
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "topic": "agentic AI",
                "draft_markdown": f"{claim} [S1]",
                "sources": [source.model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["verify", str(report), "--archive-dir", str(tmp_path / "archive")]
    )

    assert result.exit_code == 0, result.output
    assert "VERIFIED" in result.output


def test_verify_fails_for_a_missing_snapshot(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "topic": "agentic AI",
                "draft_markdown": "Agents use tools to accomplish goals beyond a completion. [S1]",
                "sources": [
                    {
                        "id": "S1",
                        "url": "https://example.com/a",
                        "title": "A",
                        "content_hash": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["verify", str(report), "--archive-dir", str(tmp_path / "archive")]
    )

    assert result.exit_code == 1
    assert "NOT VERIFIED" in result.output


def test_delete_run_cli_removes_only_the_named_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MARS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MARS_DEAD_LETTER_DIR", str(tmp_path / "dead"))
    monkeypatch.setenv("MARS_OBSERVABILITY_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("MARS_APPROVAL_DIR", str(tmp_path / "approvals"))
    run_id = "run-delete-me"
    owned = tmp_path / f"topic-{run_id}.md"
    other = tmp_path / "topic-keep.md"
    owned.write_text("report", encoding="utf-8")
    other.write_text("keep", encoding="utf-8")
    state = tmp_path / "state" / f"{run_id}.state.json"
    state.parent.mkdir(parents=True)
    state.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        ["delete-run", "--run-id", run_id, "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert not owned.exists()
    assert not state.exists()
    assert other.exists()


def test_disclosure_cli_prints_data_flows() -> None:
    result = runner.invoke(app, ["disclosure"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["llm_provider"] == "OpenRouter"
    assert payload["data_sent"]
    assert "openrouter_api_key" not in json.dumps(payload)
