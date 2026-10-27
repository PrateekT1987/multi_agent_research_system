"""Immutable, content-addressed evidence snapshots.

A research report is only re-verifiable if the text it was based on still
exists.  Search snippets change, pages disappear and publishers rewrite
articles, so Mars archives the material it actually saw.  Snapshots are
addressed by their SHA-256 digest, written atomically, and made read-only after
the first write.  The same digest is recorded on the :class:`~mars.schemas.Source`
so a report can be checked months later without touching the network.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .storage import atomic_write_json, atomic_write_text

__all__ = ["ArchiveRecord", "EvidenceArchive"]


def _utcnow() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class ArchiveRecord:
    """Metadata for one immutable snapshot."""

    content_hash: str
    path: str
    url: str
    final_url: str
    title: str
    content_type: str
    http_status: int
    retrieved_at: str
    bytes: int
    provider: str
    truncated: bool = False
    injection_flags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "path": self.path,
            "url": self.url,
            "final_url": self.final_url,
            "title": self.title,
            "content_type": self.content_type,
            "http_status": self.http_status,
            "retrieved_at": self.retrieved_at,
            "bytes": self.bytes,
            "provider": self.provider,
            "truncated": self.truncated,
            "injection_flags": list(self.injection_flags),
        }


class EvidenceArchive:
    """A write-once snapshot store rooted at ``root``.

    ``root=None`` disables archiving (used by lightweight unit tests); the
    caller then gets ``None`` and the source keeps only its snippet.
    """

    def __init__(
        self,
        root: Path | str | None,
        *,
        enabled: bool = True,
        max_chars: int = 2_000_000,
    ) -> None:
        self.root = Path(root) if root else None
        self.enabled = bool(enabled and self.root is not None)
        self.max_chars = max(1_000, int(max_chars))

    # ------------------------------------------------------------------ paths
    def _text_path(self, digest: str) -> Path:
        assert self.root is not None
        return self.root / digest[:2] / f"{digest}.txt"

    def _meta_path(self, digest: str) -> Path:
        assert self.root is not None
        return self.root / digest[:2] / f"{digest}.json"

    # ------------------------------------------------------------------ store
    def store(
        self,
        text: str,
        *,
        url: str,
        title: str = "",
        content_type: str = "",
        http_status: int = 0,
        final_url: str = "",
        provider: str = "",
        retrieved_at: str | None = None,
        injection_flags: tuple[str, ...] | list[str] = (),
    ) -> ArchiveRecord | None:
        """Archive ``text`` and return its immutable record.

        Re-archiving identical content is a no-op: the first snapshot wins.
        """
        if not self.enabled or not text:
            return None
        original = text
        truncated = len(original) > self.max_chars
        if truncated:
            text = original[: self.max_chars]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        text_path = self._text_path(digest)
        meta_path = self._meta_path(digest)
        try:
            if not text_path.is_file():
                atomic_write_text(text_path, text)
                try:
                    os.chmod(text_path, 0o444)
                except OSError:
                    pass
            if not meta_path.is_file():
                record = ArchiveRecord(
                    content_hash=digest,
                    path=str(text_path),
                    url=url,
                    final_url=final_url or url,
                    title=title,
                    content_type=content_type,
                    http_status=int(http_status or 0),
                    retrieved_at=retrieved_at or _utcnow(),
                    bytes=len(text.encode("utf-8")),
                    provider=provider,
                    truncated=truncated,
                    injection_flags=tuple(dict.fromkeys(injection_flags)),
                )
                atomic_write_json(meta_path, {"schema_version": 1, **record.as_dict()})
                try:
                    os.chmod(meta_path, 0o444)
                except OSError:
                    pass
        except OSError:
            return None
        return self.metadata(digest)

    # ------------------------------------------------------------------- read
    def load(self, content_hash: str) -> str | None:
        """Return the archived text for ``content_hash``, if present."""
        if not self.enabled or not content_hash:
            return None
        path = self._text_path(content_hash)
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def metadata(self, content_hash: str) -> ArchiveRecord | None:
        if not self.enabled or not content_hash:
            return None
        try:
            payload = json.loads(self._meta_path(content_hash).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return ArchiveRecord(
                content_hash=payload["content_hash"],
                path=payload["path"],
                url=payload.get("url", ""),
                final_url=payload.get("final_url", ""),
                title=payload.get("title", ""),
                content_type=payload.get("content_type", ""),
                http_status=int(payload.get("http_status", 0) or 0),
                retrieved_at=payload.get("retrieved_at", ""),
                bytes=int(payload.get("bytes", 0) or 0),
                provider=payload.get("provider", ""),
                truncated=bool(payload.get("truncated", False)),
                injection_flags=tuple(payload.get("injection_flags", []) or []),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def verify(self, content_hash: str) -> bool:
        """True when the archived bytes still hash to their address."""
        text = self.load(content_hash)
        if text is None:
            return False
        return hashlib.sha256(text.encode("utf-8")).hexdigest() == content_hash

    def resolve(self, source: Any) -> str | None:
        """Load the best available text for a source (snapshot, else snippet)."""
        content_hash = getattr(source, "content_hash", "") or ""
        text = self.load(content_hash) if content_hash else None
        if text:
            return text
        return getattr(source, "snippet", "") or None

    def stats(self) -> dict[str, int]:
        if not self.enabled or self.root is None:
            return {"entries": 0, "bytes": 0}
        entries = 0
        total = 0
        if self.root.is_dir():
            for path in self.root.rglob("*.txt"):
                try:
                    total += path.stat().st_size
                    entries += 1
                except OSError:
                    continue
        return {"entries": entries, "bytes": total}
