"""P2: immutable, content-addressed evidence snapshots."""

from __future__ import annotations

import os
import stat

from mars.archive import EvidenceArchive
from mars.evidence import EvidenceStore


def test_store_writes_content_addressed_snapshot(tmp_path) -> None:
    archive = EvidenceArchive(tmp_path)
    record = archive.store(
        "Agents use tools to accomplish goals.",
        url="https://example.com/a",
        title="Agents",
        content_type="text/html",
        http_status=200,
        provider="fetch",
    )

    assert record is not None
    assert len(record.content_hash) == 64
    assert record.path.endswith(".txt")
    assert archive.verify(record.content_hash)
    assert archive.load(record.content_hash) == "Agents use tools to accomplish goals."
    assert archive.stats()["entries"] == 1


def test_archiving_is_write_once(tmp_path) -> None:
    archive = EvidenceArchive(tmp_path)
    first = archive.store("original", url="https://example.com/a", title="Original")
    second = archive.store("original", url="https://example.com/a", title="Changed")

    assert first is not None and second is not None
    assert first.content_hash == second.content_hash
    # The first snapshot and its metadata win; re-archiving is a no-op.
    assert second.title == "Original"
    assert archive.load(first.content_hash) == "original"


def test_snapshot_is_read_only(tmp_path) -> None:
    archive = EvidenceArchive(tmp_path)
    record = archive.store("immutable", url="https://example.com/a")
    assert record is not None
    mode = stat.S_IMODE(os.stat(record.path).st_mode)

    assert mode & 0o222 == 0, "archived snapshot must not be writable"


def test_tampering_is_detected(tmp_path) -> None:
    archive = EvidenceArchive(tmp_path)
    record = archive.store("trusted text", url="https://example.com/a")
    assert record is not None
    os.chmod(record.path, 0o644)
    with open(record.path, "w", encoding="utf-8") as handle:
        handle.write("tampered text")

    assert archive.verify(record.content_hash) is False


def test_truncation_is_recorded(tmp_path) -> None:
    archive = EvidenceArchive(tmp_path, max_chars=1_000)
    record = archive.store("x" * 5_000, url="https://example.com/a")

    assert record is not None
    assert record.truncated is True
    assert record.bytes == 1_000


def test_disabled_archive_returns_none(tmp_path) -> None:
    archive = EvidenceArchive(tmp_path, enabled=False)

    assert archive.store("text", url="https://example.com") is None
    assert archive.stats() == {"entries": 0, "bytes": 0}


def test_evidence_store_records_provenance(tmp_path) -> None:
    archive = EvidenceArchive(tmp_path)
    store = EvidenceStore(archive=archive)
    source = store.register(
        "https://example.com/page",
        "A page",
        "snippet",
        kind="web",
        provider="fetch",
        content="The archived body of the page.",
        content_type="text/html",
        http_status=200,
        final_url="https://example.com/page",
        quality=0.8,
        license="CC-BY",
        injection_flags=("instruction_override",),
    )

    assert source.content_hash
    assert source.archive_path
    assert source.final_url == "https://example.com/page"
    assert source.content_type == "text/html"
    assert source.http_status == 200
    assert source.quality == 0.8
    assert source.license == "CC-BY"
    assert source.injection_flags == ["instruction_override"]
    assert archive.verify(source.content_hash)
    assert archive.resolve(source) == "The archived body of the page."


def test_archive_resolve_falls_back_to_snippet(tmp_path) -> None:
    archive = EvidenceArchive(tmp_path)
    store = EvidenceStore(archive=archive)
    source = store.register("https://example.com/x", "X", "just a snippet")

    assert source.content_hash == ""
    assert archive.resolve(source) == "just a snippet"
