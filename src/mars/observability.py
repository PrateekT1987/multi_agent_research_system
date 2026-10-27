"""Structured, local observability for Mars runs.

Events are JSONL so they can be tailed locally or ingested by OpenTelemetry,
Datadog, ELK, or any log pipeline without making observability a new runtime
dependency or a new failure mode.  Every event carries the run's trace id and a
per-span id, and the same record is mirrored into Python's structured logger so
application logs and traces can be correlated.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from .storage import append_jsonl

__all__ = ["RunObserver", "configure_logging", "send_alert"]

_LOGGER = logging.getLogger("mars.observability")


class RunObserver:
    """Thread-safe trace/event emitter scoped to one research run."""

    def __init__(
        self,
        directory: Path,
        trace_id: str,
        *,
        sample_rate: float = 1.0,
        otel_endpoint: str | None = None,
        service_name: str = "mars",
        service_version: str = "0.3.0",
    ) -> None:
        self.trace_id = trace_id
        self.path = Path(directory) / f"{trace_id}.jsonl"
        self.sample_rate = min(1.0, max(0.0, float(sample_rate)))
        self.sampled = self.sample_rate >= 1.0 or random.random() < self.sample_rate
        self.otel_endpoint = otel_endpoint or None
        self.service_name = service_name
        self.service_version = service_version
        self.emitted = 0
        self.write_errors = 0
        self._warned = False
        self._lock = threading.Lock()

    def emit(self, name: str, *, status: str = "ok", **attributes: Any) -> None:
        if not self.sampled:
            return
        event = {
            "timestamp": time.time(),
            "trace_id": self.trace_id,
            "span_id": uuid4().hex[:16],
            "parent_span_id": attributes.pop("parent_span_id", None),
            "name": name,
            "status": status,
            "service": {"name": self.service_name, "version": self.service_version},
            "attributes": attributes,
        }
        with self._lock:
            self.emitted += 1
        try:
            append_jsonl(self.path, event)
        except OSError:
            # Telemetry must never block research, but silent loss is a bug.
            with self._lock:
                self.write_errors += 1
                first = not self._warned
                self._warned = True
            if first:
                print(
                    f"mars: warning: could not write trace event to {self.path}",
                    file=sys.stderr,
                )
        _LOGGER.info(
            name,
            extra={
                "trace_id": self.trace_id,
                "span_id": event["span_id"],
                "mars_status": status,
                "mars_attributes": attributes,
            },
        )
        if self.otel_endpoint:
            self._export(event)

    def _export(self, event: dict[str, Any]) -> None:
        """Best-effort OTLP/HTTP JSON export; failures never affect a run."""
        try:
            import httpx

            httpx.post(self.otel_endpoint, json=event, timeout=2.0)
        except Exception:  # pragma: no cover - exporter is optional
            return

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        started = time.perf_counter()
        self.emit(f"{name}.started", **attributes)
        try:
            yield
        except Exception as exc:
            self.emit(
                f"{name}.finished",
                status="error",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                error_type=type(exc).__name__,
                error=str(exc)[:500],
                **attributes,
            )
            raise
        else:
            self.emit(
                f"{name}.finished",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                **attributes,
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "path": str(self.path),
            "sampled": self.sampled,
            "sample_rate": self.sample_rate,
            "emitted": self.emitted,
            "write_errors": self.write_errors,
            "otel_endpoint": self.otel_endpoint,
        }


def configure_logging(level: str | None = None) -> None:
    """Configure a compact structured logger for CLI/server deployments."""
    resolved = (level or os.environ.get("MARS_LOG_LEVEL", "INFO")).upper()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=resolved,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    logging.getLogger("mars").setLevel(resolved)


def send_alert(
    webhook_url: str | None,
    kind: str,
    message: str,
    **fields: Any,
) -> bool:
    """Best-effort operator alert; never raises and never blocks a run.

    ``MARS_ALERT_WEBHOOK_URL`` accepts any endpoint that can receive a JSON
    POST (Slack incoming webhook, PagerDuty Events proxy, an internal relay).
    """
    if not webhook_url:
        return False
    payload = {
        "service": "mars",
        "kind": kind,
        "message": message,
        "timestamp": time.time(),
        "fields": fields,
    }
    try:
        import httpx

        response = httpx.post(webhook_url, json=payload, timeout=2.0)
        return response.status_code < 400
    except Exception:  # pragma: no cover - alerting must not fail the run
        _LOGGER.warning("alert delivery failed for %s", kind, exc_info=True)
        return False
