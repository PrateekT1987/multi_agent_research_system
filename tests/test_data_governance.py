"""P2: data-governance disclosure and run deletion."""

from __future__ import annotations

from mars.config import build_settings
from mars.data_governance import delete_run, disclosure, run_artifacts


def _settings(tmp_path, **overrides):
    base = {
        "llm_provider": "openrouter",
        "openrouter_api_key": "sk-test",
        "cache_dir": tmp_path / "cache",
        "output_dir": tmp_path / "out",
        "dead_letter_dir": tmp_path / "dead",
        "observability_dir": tmp_path / "traces",
        "approval_dir": tmp_path / "approvals",
        "archive_dir": tmp_path / "archive",
        "retention_days": 14,
        "env_file": None,
    }
    base.update(overrides)
    return build_settings(**base)


def test_disclosure_describes_external_data_flows(tmp_path) -> None:
    info = disclosure(_settings(tmp_path))

    assert info["domain"] == "Agentic AI"
    assert info["llm_provider"] == "OpenRouter"
    assert info["llm_base_url"]
    assert info["models"]["strong"]
    assert "the research topic" in info["data_sent"]
    assert info["redaction"]["secrets"] is True
    assert info["redaction"]["personal_data"] is True
    assert info["at_rest_encryption"] is False
    assert info["retention_days"] == 14
    assert "delete-run" in info["note"]


def test_disclosure_reports_encryption_and_pii_toggle(tmp_path) -> None:
    info = disclosure(
        _settings(
            tmp_path,
            redact_pii=False,
            encryption_key="encryption-key",
            encrypt_state=True,
        )
    )

    assert info["at_rest_encryption"] is True
    assert info["redaction"]["personal_data"] is False


def test_run_artifacts_finds_every_owned_file(tmp_path) -> None:
    settings = _settings(tmp_path)
    run_id = "run-test-1"
    paths = [
        settings.output_dir / f"topic-{run_id}.md",
        settings.output_dir / f"topic-{run_id}-run.json",
        settings.state_dir / f"{run_id}.state.json",
        settings.dead_letter_dir / f"{run_id}.jsonl",
        settings.approval_dir / f"{run_id}-draft.request.json",
        settings.observability_dir / f"{run_id}.trace.jsonl",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    unrelated = settings.output_dir / "topic-other-run.md"
    unrelated.write_text("keep", encoding="utf-8")

    found = {path.name for path in run_artifacts(settings, run_id)}

    assert {path.name for path in paths} <= found
    assert unrelated.name not in found


def test_delete_run_dry_run_keeps_everything(tmp_path) -> None:
    settings = _settings(tmp_path)
    run_id = "run-test-2"
    report = settings.output_dir / f"topic-{run_id}.md"
    report.parent.mkdir(parents=True)
    report.write_text("report", encoding="utf-8")

    deletion = delete_run(settings, run_id, dry_run=True)

    assert report.exists()
    assert deletion.dry_run is True
    assert str(report) in deletion.removed


def test_delete_run_removes_owned_files_and_keeps_others(tmp_path) -> None:
    settings = _settings(tmp_path)
    run_id = "run-test-3"
    owned = settings.output_dir / f"topic-{run_id}.md"
    other = settings.output_dir / "topic-other.md"
    owned.parent.mkdir(parents=True)
    owned.write_text("report", encoding="utf-8")
    other.write_text("keep", encoding="utf-8")
    archive_file = settings.archive_dir / "aa" / "snapshot.txt"
    archive_file.parent.mkdir(parents=True)
    archive_file.write_text("shared", encoding="utf-8")

    deletion = delete_run(settings, run_id)

    assert not owned.exists()
    assert other.exists()
    assert archive_file.exists(), "the shared evidence archive must not be deleted"
    assert not deletion.errors
