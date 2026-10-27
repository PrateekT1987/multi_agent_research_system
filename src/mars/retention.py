"""Retention and disk-growth control.

Mars writes reports, traces, dead letters, run states and approvals forever if
nobody prunes them.  This module centralises the policy so the CLI, a cron job
and an operator all use the same, testable rules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .storage import file_age_seconds

__all__ = ["PruneReport", "prune_all", "prune_cache_dir"]


@dataclass(slots=True)
class PruneReport:
    removed: list[str] = field(default_factory=list)
    kept: int = 0
    bytes_freed: int = 0
    dry_run: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "removed": len(self.removed),
            "kept": self.kept,
            "bytes_freed": self.bytes_freed,
            "dry_run": self.dry_run,
            "paths": self.removed,
        }


def _prune_glob(
    root: Path,
    pattern: str,
    *,
    max_age_seconds: float,
    report: PruneReport,
) -> None:
    if not root.is_dir():
        return
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        if file_age_seconds(path) <= max_age_seconds:
            report.kept += 1
            continue
        try:
            size = path.stat().st_size
            if not report.dry_run:
                path.unlink()
            report.removed.append(str(path))
            report.bytes_freed += size
        except OSError:
            continue


def prune_cache_dir(
    cache_dir: Path,
    *,
    max_bytes: int,
    max_entries: int,
    dry_run: bool = False,
) -> PruneReport:
    """Evict the oldest HTTP cache entries until both caps are respected."""
    report = PruneReport(dry_run=dry_run)
    root = Path(cache_dir) / "http"
    if not root.is_dir():
        return report
    entries: list[tuple[float, int, Path]] = []
    total = 0
    for path in root.rglob("*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
        total += stat.st_size
    entries.sort(key=lambda item: item[0])
    index = 0
    while entries and (len(entries) > max_entries or total > max_bytes):
        if index >= len(entries):
            break
        _, size, path = entries[index]
        try:
            if not dry_run:
                path.unlink()
            report.removed.append(str(path))
            report.bytes_freed += size
            total -= size
            entries.pop(index)
        except OSError:
            index += 1
    report.kept = len(entries)
    return report


def prune_all(settings: Settings, *, days: int | None = None, dry_run: bool = False) -> PruneReport:
    """Apply the configured retention window to every durable artefact."""
    window = settings.retention_days if days is None else days
    max_age = max(0, window) * 86_400
    report = PruneReport(dry_run=dry_run)

    for pattern in ("*-run.json", "*-sources.json", "*-plan.json", "*.md", "*.json"):
        _prune_glob(settings.output_dir, pattern, max_age_seconds=max_age, report=report)
    for pattern in ("*.jsonl",):
        _prune_glob(settings.observability_dir, pattern, max_age_seconds=max_age, report=report)
        _prune_glob(settings.dead_letter_dir, pattern, max_age_seconds=max_age, report=report)
    for pattern in ("*.state.json",):
        _prune_glob(settings.state_dir, pattern, max_age_seconds=max_age, report=report)
    for pattern in ("*.request.json", "*.approved.json", "*.rejected.json", "*.payload.json"):
        _prune_glob(settings.approval_dir, pattern, max_age_seconds=max_age, report=report)

    cache_report = prune_cache_dir(
        settings.cache_dir,
        max_bytes=settings.cache_max_bytes,
        max_entries=settings.cache_max_entries,
        dry_run=dry_run,
    )
    report.removed.extend(cache_report.removed)
    report.bytes_freed += cache_report.bytes_freed
    report.kept += cache_report.kept
    return report


def now() -> float:
    """Small indirection that makes retention tests deterministic."""
    return time.time()
