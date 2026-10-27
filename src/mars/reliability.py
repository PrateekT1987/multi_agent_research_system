"""Run-scoped failure containment for external model calls.

The objects in this module are deliberately synchronous and thread-safe: one
``ResearchContext`` is shared by every parallel track, so an upstream outage
opens one breaker for the entire fleet rather than multiplying retries.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CircuitOpenError",
    "ProviderUnavailableError",
    "ProviderResponseError",
    "CallBudgetExceeded",
    "SharedCircuitBreaker",
    "get_shared_circuit_breaker",
    "breaker_snapshots",
    "CallBudget",
    "DeadLetterQueue",
    "RunStateStore",
    "WatchdogExpired",
    "RunWatchdog",
]

_BREAKER_REGISTRY: dict[str, SharedCircuitBreaker] = {}
_BREAKER_LOCK = threading.Lock()


def get_shared_circuit_breaker(
    key: str,
    failure_threshold: int = 4,
    recovery_seconds: float = 30.0,
) -> SharedCircuitBreaker:
    """Return the process-wide breaker for ``key``.

    A per-run breaker protects one run from an outage; a process-wide breaker
    protects the whole worker (and every concurrent run inside it) from
    multiplying retries against the same failing provider.
    """
    with _BREAKER_LOCK:
        breaker = _BREAKER_REGISTRY.get(key)
        if breaker is None:
            breaker = SharedCircuitBreaker(
                failure_threshold, recovery_seconds, name=key
            )
            _BREAKER_REGISTRY[key] = breaker
        return breaker


def breaker_snapshots() -> list[dict[str, Any]]:
    """Snapshot every process-wide breaker for health/metrics output."""
    with _BREAKER_LOCK:
        breakers = list(_BREAKER_REGISTRY.values())
    return [breaker.snapshot() for breaker in breakers]


class CircuitOpenError(RuntimeError):
    """An upstream dependency is in its shared recovery window."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        #: Seconds until the breaker may admit a probe, when known.  Callers
        #: use this to decide whether waiting is cheaper than failing.
        self.retry_after = retry_after


class ProviderUnavailableError(RuntimeError):
    """No agent in a run could reach the model provider.

    Raised only when the run has no usable work left: every research track
    failed before producing evidence.  The CLI turns it into a one-line,
    actionable message instead of a provider traceback.
    """


class ProviderResponseError(RuntimeError):
    """A provider answered with an error envelope instead of a completion.

    OpenRouter can return HTTP 200 with ``{"id": ..., "error": {...}}`` when an
    upstream provider is saturated.  The SDK then fails to unmarshal the body
    into a chat completion, which used to surface as an opaque Pydantic
    ``Unmarshaller`` traceback.  This type carries the provider's own code so
    retries, the breaker and the dead-letter record can classify it.
    """

    def __init__(self, code: str, message: str = "", *, provider: str = "openrouter") -> None:
        self.code = code
        self.message = message
        self.provider = provider
        detail = f": {message}" if message else ""
        super().__init__(f"{provider} returned {code}{detail}")


class CallBudgetExceeded(RuntimeError):
    """The run has reached its configured external-call budget."""


class WatchdogExpired(RuntimeError):
    """A bounded pipeline stage exceeded its operational deadline."""


class SharedCircuitBreaker:
    """Closed/open/half-open circuit breaker shared by every agent in one run."""

    def __init__(
        self,
        failure_threshold: int = 4,
        recovery_seconds: float = 30.0,
        *,
        name: str = "default",
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_seconds = max(0.0, recovery_seconds)
        self.name = name
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False
        self._cond = threading.Condition()

    def before_call(self, *, timeout: float = 0.0) -> bool:
        """Permit a call or fail fast while the dependency recovers.

        ``timeout=0`` (the default) preserves fail-fast behaviour: an open
        breaker raises immediately.  A positive timeout lets the parallel
        agents of one run wait out a single shared recovery window instead of
        turning a short provider blip into a fleet-wide failure.  While the
        breaker is open exactly one caller becomes the half-open probe; the
        others wait for its outcome.

        Returns ``True`` when the caller was admitted as the half-open probe,
        so it can release the probe if it ends without a provider verdict.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while True:
                if self._opened_at is None:
                    return False
                elapsed = time.monotonic() - self._opened_at
                remaining = max(0.0, self.recovery_seconds - elapsed)
                if remaining <= 0.0 and not self._probe_in_flight:
                    self._probe_in_flight = True
                    return True
                now = time.monotonic()
                if now >= deadline:
                    if remaining <= 0.0:
                        raise CircuitOpenError(
                            "LLM circuit is half-open; recovery probe is in flight",
                            retry_after=self.recovery_seconds,
                        )
                    raise CircuitOpenError(
                        f"LLM circuit is open; retry after {remaining:.1f}s",
                        retry_after=remaining,
                    )
                if self._probe_in_flight:
                    # The probe owner decides the next state; wait for it.
                    wait_for = deadline - now
                else:
                    wait_for = min(remaining, deadline - now)
                self._cond.wait(max(0.01, wait_for))

    def release_probe(self) -> None:
        """Release a half-open probe that ended without a provider verdict.

        A non-retryable client error (a malformed request, say) says nothing
        about provider availability.  It must not count as a failure, but it
        must also not leave the probe flag set, or every other waiter would
        block until its timeout.
        """
        with self._cond:
            if self._probe_in_flight:
                self._probe_in_flight = False
                self._cond.notify_all()

    def record_success(self) -> None:
        with self._cond:
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False
            self._cond.notify_all()

    def record_failure(self) -> None:
        with self._cond:
            self._failures += 1
            self._probe_in_flight = False
            if self._opened_at is not None or self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()
            self._cond.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            state = "open" if self._opened_at is not None else "closed"
            if state == "open" and self._probe_in_flight:
                state = "half_open"
            return {
                "name": self.name,
                "state": state,
                "failures": self._failures,
                "failure_threshold": self.failure_threshold,
            }


class CallBudget:
    """Hard upper bound on provider requests made by an entire research run."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max(1, max_calls)
        self.calls = 0
        self._lock = threading.Lock()

    def reserve(self) -> None:
        with self._lock:
            if self.calls >= self.max_calls:
                raise CallBudgetExceeded(f"LLM call budget exhausted ({self.max_calls} calls)")
            self.calls += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"calls_reserved": self.calls, "max_calls": self.max_calls}


@dataclass(frozen=True, slots=True)
class _DeadLetter:
    run_id: str
    stage: str
    error_type: str
    error: str
    payload: dict[str, Any]
    created_at: float
    schema_version: int = 1


class DeadLetterQueue:
    """Append-only JSONL queue for failed, manually recoverable work items."""

    def __init__(self, directory: Path, run_id: str, *, cipher: Any = None) -> None:
        self.path = Path(directory) / f"{run_id}.jsonl"
        self.run_id = run_id
        self.cipher = cipher
        self._lock = threading.Lock()

    def enqueue(self, stage: str, error: Exception, payload: dict[str, Any] | None = None) -> Path:
        item = _DeadLetter(
            run_id=self.run_id,
            stage=stage,
            error_type=type(error).__name__,
            error=str(error)[:2000],
            payload=payload or {},
            created_at=time.time(),
        )
        if self.cipher is not None and getattr(self.cipher, "enabled", False):
            encoded = self.cipher.encrypt_json(asdict(item)) + "\n"
        else:
            encoded = json.dumps(asdict(item), default=str, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        return self.path

    def read_items(self) -> list[dict[str, Any]]:
        """Read every dead-letter record for this run."""
        with self._lock:
            if not self.path.is_file():
                return []
            items: list[dict[str, Any]] = []
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    if self.cipher is not None and self.cipher.is_encrypted(line):
                        items.append(self.cipher.decrypt_json(line))
                    else:
                        items.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
            return items

    def count(self) -> int:
        return len(self.read_items())

    def mark_replayed(self, *, replay_run_id: str) -> Path:
        """Record that a replay attempt consumed the current queue."""
        marker = self.path.with_suffix(".replayed.json")
        payload = {
            "run_id": self.run_id,
            "replay_run_id": replay_run_id,
            "replayed_at": time.time(),
            "items": self.count(),
        }
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, marker)
        return marker


class RunStateStore:
    """Atomic checkpoint journal for restart diagnosis and safe human review.

    This is intentionally a small file-backed state machine rather than an
    in-memory dict: a process crash must not erase which stages completed or
    which work needs replaying from the DLQ.
    """

    def __init__(self, directory: Path, run_id: str, *, cipher: Any = None) -> None:
        self.path = Path(directory) / f"{run_id}.state.json"
        self.run_id = run_id
        self.cipher = cipher
        self._lock = threading.Lock()
        existing = self._read_existing()
        if existing is not None:
            self._state = existing
        else:
            self._state = {
                "schema_version": 1,
                "run_id": run_id,
                "status": "running",
                "created_at": time.time(),
                "updated_at": time.time(),
                "stages": {},
            }
            self._write()

    def _read_existing(self) -> dict[str, Any] | None:
        try:
            text = self.path.read_text(encoding="utf-8")
            if self.cipher is not None and self.cipher.is_encrypted(text):
                data = self.cipher.decrypt_json(text)
            else:
                data = json.loads(text)
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("run_id") != self.run_id:
            return None
        data.setdefault("schema_version", 1)
        data.setdefault("stages", {})
        return data

    @classmethod
    def load(cls, directory: Path, run_id: str, *, cipher: Any = None) -> RunStateStore | None:
        """Load an existing state journal without creating a new one."""
        path = Path(directory) / f"{run_id}.state.json"
        if not path.is_file():
            return None
        store = cls(directory, run_id, cipher=cipher)
        return store if store.path.is_file() else None

    def read(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    @property
    def status(self) -> str:
        with self._lock:
            return str(self._state.get("status", "unknown"))

    def stage_summary(self, stage: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._state.get("stages", {}).get(stage)
            if not entry:
                return None
            summary = entry.get("summary")
            return summary if isinstance(summary, dict) else None

    def stage_status(self, stage: str) -> str | None:
        with self._lock:
            entry = self._state.get("stages", {}).get(stage)
            return str(entry.get("status")) if entry else None

    def transition(
        self, stage: str, status: str, *, summary: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            stages = self._state["stages"]
            entry = dict(stages.get(stage, {}))
            entry.update({"status": status, "updated_at": time.time()})
            if summary is not None:
                entry["summary"] = summary
            stages[stage] = entry
            self._state["updated_at"] = time.time()
            self._write()

    def finish(self, status: str, *, summary: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._state["status"] = status
            self._state["updated_at"] = time.time()
            if summary is not None:
                self._state["summary"] = summary
            self._write()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            # JSON round-trip returns a safe, detached snapshot.
            return json.loads(json.dumps(self._state))

    def _write(self) -> None:
        from .storage import atomic_write_json, atomic_write_text

        if self.cipher is not None and self.cipher.enabled:
            atomic_write_text(self.path, self.cipher.encrypt_json(self._state))
        else:
            atomic_write_json(self.path, self._state)


class RunWatchdog:
    """Cooperative deadline guard for a whole run and its individual tracks."""

    def __init__(self, max_run_seconds: float, max_track_seconds: float) -> None:
        self.max_run_seconds = max(1.0, max_run_seconds)
        self.max_track_seconds = max(1.0, max_track_seconds)
        self.started = time.monotonic()
        self._stop = threading.Event()

    def request_stop(self) -> None:
        """Ask every cooperating stage to unwind at its next checkpoint."""
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def remaining_run_seconds(self) -> float:
        return max(0.0, self.max_run_seconds - (time.monotonic() - self.started))

    def check_run(self) -> None:
        if self._stop.is_set():
            raise WatchdogExpired("graceful shutdown requested")
        elapsed = time.monotonic() - self.started
        if elapsed > self.max_run_seconds:
            raise WatchdogExpired(f"run deadline exceeded ({self.max_run_seconds:.0f}s)")

    def check_track(self, track_started: float, track: str) -> None:
        self.check_run()
        elapsed = time.monotonic() - track_started
        if elapsed > self.max_track_seconds:
            raise WatchdogExpired(f"track '{track}' deadline exceeded ({self.max_track_seconds:.0f}s)")
