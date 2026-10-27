"""P2: offline re-verification of a persisted report."""

from __future__ import annotations

import json
import os

from mars.archive import EvidenceArchive
from mars.schemas import Source
from mars.verification import verification_summary, verify_report_file

_CLAIM = "Agents use tools to accomplish goals beyond a single completion."


def _write_report(tmp_path, *, body: str = _CLAIM, archived: bool = True) -> tuple[object, EvidenceArchive]:
    archive = EvidenceArchive(tmp_path / "archive")
    source = Source(
        id="S1",
        url="https://example.com/agents",
        title="Agents",
        snippet=_CLAIM,
    )
    if archived:
        record = archive.store(_CLAIM, url=source.url, title=source.title)
        assert record is not None
        source.content_hash = record.content_hash
        source.archive_path = record.path
    report = {
        "topic": "agentic AI",
        "generated_at": "2026-10-26T00:00:00+00:00",
        "draft_markdown": f"{body} [S1]",
        "sources": [source.model_dump(mode="json")],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path, archive


def test_verify_report_passes_for_intact_snapshots(tmp_path) -> None:
    path, archive = _write_report(tmp_path)

    result = verify_report_file(path, archive.root)

    assert result["verified"] is True
    assert result["sources"] == 1
    assert result["archived_sources"] == 1
    assert result["missing_or_corrupt_snapshots"] == []
    assert result["entailment"]["supported"] == 1
    assert "VERIFIED" in verification_summary(result)


def test_verify_report_fails_when_a_snapshot_is_tampered(tmp_path) -> None:
    path, archive = _write_report(tmp_path)
    snapshot = next(archive.root.rglob("*.txt"))
    os.chmod(snapshot, 0o644)
    snapshot.write_text("Completely unrelated bananas text.", encoding="utf-8")

    result = verify_report_file(path, archive.root)

    assert result["verified"] is False
    assert result["missing_or_corrupt_snapshots"] == ["S1"]
    assert "NOT VERIFIED" in verification_summary(result)


def test_verify_report_fails_without_an_archive(tmp_path) -> None:
    path, _archive = _write_report(tmp_path)

    result = verify_report_file(path, None)

    assert result["verified"] is False
    assert result["missing_or_corrupt_snapshots"] == ["S1"]


def test_unarchived_sources_are_not_verifiable(tmp_path) -> None:
    path, archive = _write_report(tmp_path, archived=False)

    result = verify_report_file(path, archive.root)

    assert result["verified"] is False
    assert result["unarchived_sources"] == ["S1"]
    assert result["archived_sources"] == 0


def test_verify_report_rejects_a_non_report_file(tmp_path) -> None:
    path = tmp_path / "not-a-report.json"
    path.write_text('{"hello": "world"}', encoding="utf-8")

    try:
        verify_report_file(path, tmp_path / "archive")
    except ValueError as exc:
        assert "not a Mars report" in str(exc)
    else:  # pragma: no cover - the assertion above must fire
        raise AssertionError("expected a ValueError")
