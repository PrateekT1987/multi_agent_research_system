"""Offline re-verification of a persisted report.

A report is only trustworthy if a later reader can check it without trusting
the original run.  This module loads a ``*-run.json`` / ``*-<run>.json`` report,
rehydrates the source registry, and re-runs the deterministic entailment audit
against the immutable evidence archive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .archive import EvidenceArchive
from .entailment import audit_text
from .evidence import EvidenceStore
from .schemas import Source

__all__ = ["verify_report_file", "verification_summary"]


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a report object")
    if "draft_markdown" in payload:
        return payload
    if isinstance(payload.get("report"), dict):
        return payload["report"]
    raise ValueError(f"{path} is not a Mars report artefact")


def verify_report_file(
    report_path: str | Path,
    archive_dir: str | Path | None,
    *,
    supported_at: float = 0.55,
    partial_at: float = 0.30,
) -> dict[str, Any]:
    """Re-audit a persisted report and verify every archived source hash."""
    path = Path(report_path)
    report = _load_report(path)
    archive = EvidenceArchive(archive_dir, enabled=archive_dir is not None)
    store = EvidenceStore(archive=archive)
    store.restore([Source.model_validate(item) for item in report.get("sources", [])])

    audit = audit_text(
        store,
        report.get("draft_markdown", ""),
        supported_at=supported_at,
        partial_at=partial_at,
    )
    missing = [source.id for source in store.sources() if source.content_hash and not archive.verify(source.content_hash)]
    unarchived = [source.id for source in store.sources() if not source.content_hash]
    result = {
        "report": str(path),
        "topic": report.get("topic", ""),
        "generated_at": report.get("generated_at", ""),
        "archive_dir": str(archive_dir) if archive_dir else None,
        "sources": len(store.sources()),
        "archived_sources": len(store.sources()) - len(unarchived),
        "missing_or_corrupt_snapshots": missing,
        "unarchived_sources": unarchived,
        "entailment": audit.as_dict(),
    }
    result["verified"] = bool(
        not missing
        and not unarchived
        and not audit.unsupported
        and audit.ratio >= 0.55
    )
    return result


def verification_summary(result: dict[str, Any]) -> str:
    audit = result.get("entailment", {})
    status = "VERIFIED" if result.get("verified") else "NOT VERIFIED"
    return (
        f"{status}: {audit.get('supported', 0)}/{audit.get('total', 0)} sentences supported "
        f"(ratio {audit.get('ratio', 0):.2f}); "
        f"{result.get('archived_sources', 0)}/{result.get('sources', 0)} sources archived; "
        f"{len(result.get('missing_or_corrupt_snapshots', []))} snapshot problem(s)"
    )
