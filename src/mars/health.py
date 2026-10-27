"""Health, readiness and Prometheus-style metrics for Mars deployments.

The module is deliberately dependency-free (stdlib only) so a container can
expose liveness/readiness probes without adding a web framework.  It reads the
same on-disk state that the pipeline writes, which means the probe reflects
real queue depth, cache footprint and run outcomes rather than a hard-coded
"ok".
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import ConfigError, Settings, build_settings
from .reliability import breaker_snapshots

__all__ = [
    "HealthReport",
    "health_snapshot",
    "metrics_text",
    "run_server",
]


@dataclass(slots=True)
class HealthCheck:
    name: str
    ok: bool
    detail: str
    critical: bool = False


@dataclass(slots=True)
class HealthReport:
    status: str
    ready: bool
    checks: list[HealthCheck]
    facts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ready": self.ready,
            "checks": [asdict(check) for check in self.checks],
            "facts": self.facts,
        }


def _count_files(root: Path, pattern: str) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def _size_files(root: Path, pattern: str) -> int:
    if not root.is_dir():
        return 0
    total = 0
    for path in root.rglob(pattern):
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _run_state_counts(state_dir: Path) -> dict[str, int]:
    counts = {"running": 0, "completed": 0, "failed": 0, "other": 0}
    if not state_dir.is_dir():
        return counts
    for path in state_dir.glob("*.state.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            counts["other"] += 1
            continue
        status = str(data.get("status", "other"))
        counts[status if status in counts else "other"] += 1
    return counts


def health_snapshot(settings: Settings | None = None) -> HealthReport:
    """Collect liveness/readiness facts without making network requests."""
    settings = settings or build_settings(env_file=None)
    checks: list[HealthCheck] = []

    try:
        settings.validate()
    except ConfigError as exc:
        checks.append(HealthCheck("configuration", False, str(exc), critical=True))
    else:
        checks.append(HealthCheck("configuration", True, "valid"))

    writable = os.access(settings.data_dir, os.W_OK) if settings.data_dir.exists() else False
    if not settings.data_dir.exists():
        try:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            writable = os.access(settings.data_dir, os.W_OK)
        except OSError:
            writable = False
    checks.append(
        HealthCheck("data_dir_writable", writable, str(settings.data_dir), critical=True)
    )

    output_writable = os.access(settings.output_dir, os.W_OK) if settings.output_dir.exists() else False
    if not settings.output_dir.exists():
        try:
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            output_writable = os.access(settings.output_dir, os.W_OK)
        except OSError:
            output_writable = False
    checks.append(
        HealthCheck("output_dir_writable", output_writable, str(settings.output_dir), critical=True)
    )

    state_counts = _run_state_counts(settings.state_dir)
    dlq_items = sum(
        len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
        for path in settings.dead_letter_dir.glob("*.jsonl")
    ) if settings.dead_letter_dir.is_dir() else 0
    cache_entries = _count_files(settings.cache_dir / "http", "*.json")
    cache_bytes = _size_files(settings.cache_dir / "http", "*.json")
    trace_entries = _count_files(settings.observability_dir, "*.jsonl")

    try:
        disk = shutil.disk_usage(settings.data_dir)
        disk_free = disk.free
    except OSError:
        disk_free = 0

    checks.append(
        HealthCheck(
            "cache_within_budget",
            cache_bytes <= settings.cache_max_bytes,
            f"{cache_bytes} / {settings.cache_max_bytes} bytes",
        )
    )
    checks.append(
        HealthCheck(
            "pricing_configured",
            not settings.warnings(),
            "; ".join(settings.warnings()) or "all configured models priced",
        )
    )

    breakers = breaker_snapshots()
    open_breakers = [item for item in breakers if item.get("state") == "open"]
    checks.append(
        HealthCheck(
            "llm_circuit",
            not open_breakers,
            f"{len(open_breakers)} open breaker(s)" if open_breakers else "closed",
        )
    )

    ready = all(check.ok for check in checks if check.critical)
    failed_critical = any(check.critical and not check.ok for check in checks)
    status = "unhealthy" if failed_critical else ("degraded" if not all(c.ok for c in checks) else "ok")

    return HealthReport(
        status=status,
        ready=ready,
        checks=checks,
        facts={
            "runs": state_counts,
            "dead_letter_items": dlq_items,
            "cache_entries": cache_entries,
            "cache_bytes": cache_bytes,
            "trace_files": trace_entries,
            "disk_free_bytes": disk_free,
            "breakers": breakers,
            "llm_provider": settings.llm_provider,
            "allow_network": settings.allow_network,
            "warnings": settings.warnings(),
        },
    )


def metrics_text(settings: Settings | None = None) -> str:
    """Render a Prometheus text-exposition payload."""
    report = health_snapshot(settings)
    facts = report.facts
    lines = [
        "# HELP mars_up Whether the process can serve traffic.",
        "# TYPE mars_up gauge",
        f"mars_up {1 if report.status != 'unhealthy' else 0}",
        "# HELP mars_ready Whether critical dependencies are configured.",
        "# TYPE mars_ready gauge",
        f"mars_ready {1 if report.ready else 0}",
        "# HELP mars_dead_letter_items Failed work items awaiting replay.",
        "# TYPE mars_dead_letter_items gauge",
        f"mars_dead_letter_items {facts['dead_letter_items']}",
        "# HELP mars_cache_bytes On-disk HTTP cache size.",
        "# TYPE mars_cache_bytes gauge",
        f"mars_cache_bytes {facts['cache_bytes']}",
        "# HELP mars_cache_entries On-disk HTTP cache entries.",
        "# TYPE mars_cache_entries gauge",
        f"mars_cache_entries {facts['cache_entries']}",
        "# HELP mars_disk_free_bytes Free bytes on the data filesystem.",
        "# TYPE mars_disk_free_bytes gauge",
        f"mars_disk_free_bytes {facts['disk_free_bytes']}",
        "# HELP mars_run_states Number of run state journals by status.",
        "# TYPE mars_run_states gauge",
    ]
    for status, count in facts["runs"].items():
        lines.append(f'mars_run_states{{status="{status}"}} {count}')
    lines.extend(
        [
            "# HELP mars_breaker_open Whether an LLM circuit breaker is open.",
            "# TYPE mars_breaker_open gauge",
        ]
    )
    for breaker in facts["breakers"]:
        name = str(breaker.get("name", "default")).replace('"', "")
        lines.append(
            f'mars_breaker_open{{name="{name}"}} {1 if breaker.get("state") == "open" else 0}'
        )
    return "\n".join(lines) + "\n"


class _Handler(BaseHTTPRequestHandler):
    settings: Settings

    def log_message(self, *args: object) -> None:
        return

    def _send(self, status: int, body: str, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            report = health_snapshot(self.settings)
            self._send(200 if report.status != "unhealthy" else 503, json.dumps(report.to_dict()), "application/json")
        elif path == "/readyz":
            report = health_snapshot(self.settings)
            self._send(200 if report.ready else 503, json.dumps({"ready": report.ready}), "application/json")
        elif path == "/metrics":
            self._send(200, metrics_text(self.settings), "text/plain; version=0.0.4")
        else:
            self._send(404, "not found\n", "text/plain")


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    settings: Settings | None = None,
) -> None:
    """Serve ``/healthz``, ``/readyz`` and ``/metrics`` until interrupted."""
    from .observability import configure_logging

    configure_logging()
    resolved = settings or build_settings(env_file=None)
    handler = type("BoundHealthHandler", (_Handler,), {"settings": resolved})
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
