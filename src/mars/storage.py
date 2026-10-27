"""Small, dependency-free persistence helpers.

Every durable artefact in Mars goes through these functions so that a crash
cannot leave a half-written JSON file behind, and so that filenames are unique
across concurrent runs.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "atomic_write_text",
    "atomic_write_json",
    "append_jsonl",
    "read_json",
    "unique_run_id",
    "file_age_seconds",
]

SCHEMA_VERSION = 1


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically (same-directory temp + fsync)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Serialise ``payload`` as JSON and write it atomically."""
    atomic_write_text(path, json.dumps(payload, indent=indent, default=str))


def append_jsonl(path: str | Path, payload: Any) -> None:
    """Append one JSON line and fsync it so a crash cannot lose the record."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, default=str, sort_keys=True) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: str | Path, *, default: Any = None) -> Any:
    """Read JSON, returning ``default`` when the file is missing or corrupt."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def unique_run_id(prefix: str = "run") -> str:
    """A collision-resistant run id suitable for filenames and logs."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:10]}"


def file_age_seconds(path: str | Path) -> float:
    """Age of a file in seconds, or ``inf`` when it does not exist."""
    try:
        return max(0.0, time.time() - Path(path).stat().st_mtime)
    except OSError:
        return float("inf")
