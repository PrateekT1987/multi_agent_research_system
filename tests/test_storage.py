"""Tests for atomic persistence helpers."""

from __future__ import annotations

import json
import os

import pytest

from mars import storage


def test_atomic_write_json_creates_parent_and_round_trips(tmp_path) -> None:
    target = tmp_path / "nested" / "payload.json"

    storage.atomic_write_json(target, {"b": 2, "a": 1})

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert not list(target.parent.glob(".*.tmp"))


def test_atomic_write_failure_preserves_previous_content(tmp_path, monkeypatch) -> None:
    target = tmp_path / "payload.txt"
    target.write_text("original", encoding="utf-8")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(OSError, match="disk full"):
        storage.atomic_write_text(target, "replacement")

    assert target.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".*.tmp"))


def test_read_json_returns_default_for_corrupt_files(tmp_path) -> None:
    target = tmp_path / "broken.json"
    target.write_text("{not json", encoding="utf-8")

    assert storage.read_json(target, default={"ok": False}) == {"ok": False}


def test_append_jsonl_is_parseable_line_by_line(tmp_path) -> None:
    target = tmp_path / "events.jsonl"

    storage.append_jsonl(target, {"n": 1})
    storage.append_jsonl(target, {"n": 2})

    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"n": 1}, {"n": 2}]


def test_unique_run_ids_do_not_collide() -> None:
    ids = {storage.unique_run_id() for _ in range(200)}

    assert len(ids) == 200
    assert all(run_id.startswith("run-") for run_id in ids)


def test_file_age_seconds_for_missing_file_is_infinite(tmp_path) -> None:
    assert storage.file_age_seconds(tmp_path / "missing") == float("inf")
