"""Data-governance surface: disclosure metadata and user deletion.

Two obligations are easy to miss in a research tool:

* tell the operator exactly what leaves the machine and where it goes, and
* let them delete everything a run produced, including state, dead letters,
  approvals and traces.

Both live here so the CLI, the UI and an incident-response script share one
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .domain import DOMAIN

__all__ = ["DeletionReport", "disclosure", "run_artifacts", "delete_run"]


@dataclass(slots=True)
class DeletionReport:
    run_id: str
    removed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "removed": self.removed,
            "missing": self.missing,
            "dry_run": self.dry_run,
            "errors": self.errors,
        }


def disclosure(settings: Settings) -> dict[str, Any]:
    """Describe the external data flows of a run."""
    provider = "OpenRouter" if settings.llm_provider == "openrouter" else settings.llm_provider
    return {
        "domain": DOMAIN,
        "llm_provider": provider,
        "llm_base_url": settings.openrouter_base_url,
        "models": {
            "strong": settings.model_strong,
            "fast": settings.model_fast,
            "planner": settings.model_planner,
        },
        "data_sent": [
            "the research topic",
            "agent prompts and instructions",
            "retrieved page text and search snippets",
            "recorded findings and draft report text",
        ],
        "redaction": {
            "secrets": True,
            "personal_data": settings.redact_pii,
        },
        "at_rest_encryption": bool(settings.encryption_key),
        "retention_days": settings.retention_days,
        "evidence_archive": str(settings.evidence_dir) if settings.archive_enabled else None,
        "note": (
            "Prompts are sent to the configured model provider under that provider's "
            "terms. Mars does not train on your data. Delete a run with "
            "'mars delete-run --run-id <id>'."
        ),
    }


def _artifact_patterns(settings: Settings, run_id: str) -> list[Path]:
    output = Path(settings.output_dir)
    return [
        *(output.glob(f"*-{run_id}*")),
        *(Path(settings.state_dir).glob(f"{run_id}*")),
        *(Path(settings.dead_letter_dir).glob(f"{run_id}*")),
        *(Path(settings.approval_dir).glob(f"{run_id}*")),
        *(Path(settings.observability_dir).glob(f"{run_id}*")),
    ]


def run_artifacts(settings: Settings, run_id: str) -> list[Path]:
    """Every file/directory Mars wrote for ``run_id`` (sorted, de-duplicated)."""
    seen: dict[str, Path] = {}
    for path in _artifact_patterns(settings, run_id):
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        seen.setdefault(resolved, path)
    return sorted(seen.values(), key=lambda item: str(item))


def delete_run(settings: Settings, run_id: str, *, dry_run: bool = False) -> DeletionReport:
    """Delete every artefact for ``run_id``; never touches the shared archive."""
    report = DeletionReport(run_id=run_id, dry_run=dry_run)
    for path in run_artifacts(settings, run_id):
        if not path.exists():
            report.missing.append(str(path))
            continue
        if dry_run:
            report.removed.append(str(path))
            continue
        try:
            if path.is_dir():
                import shutil

                shutil.rmtree(path)
            else:
                path.unlink()
            report.removed.append(str(path))
        except OSError as exc:
            report.errors.append(f"{path}: {exc}")
    return report
