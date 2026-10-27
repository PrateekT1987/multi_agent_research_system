"""Shared runtime context handed to every tool.

Tools need three things: the settings that shape their behaviour, the HTTP
client (so that all tools share one connection pool and one rate limiter), and
the evidence store that records what they found.  Passing them explicitly -
rather than through globals - keeps the system testable and lets independent
tracks run in parallel without interfering with each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .archive import EvidenceArchive
from .config import Settings
from .crypto import PayloadCipher
from .evidence import EvidenceStore
from .governance import CrawlerPolicy
from .net import HttpClient
from .observability import RunObserver
from .reliability import (
    CallBudget,
    DeadLetterQueue,
    RunStateStore,
    RunWatchdog,
    SharedCircuitBreaker,
    get_shared_circuit_breaker,
)
from .storage import unique_run_id

__all__ = ["ResearchContext"]


@dataclass(slots=True)
class ResearchContext:
    """Per-run container for shared services."""

    settings: Settings
    evidence: EvidenceStore = field(default_factory=EvidenceStore)
    http: HttpClient | None = None
    llm_breaker: SharedCircuitBreaker | None = None
    llm_budget: CallBudget | None = None
    dead_letters: DeadLetterQueue | None = None
    run_state: RunStateStore | None = None
    watchdog: RunWatchdog | None = None
    observer: RunObserver | None = None
    archive: EvidenceArchive | None = None
    governance: CrawlerPolicy | None = None
    cipher: PayloadCipher | None = None

    def __post_init__(self) -> None:
        run_id = self.settings.run_id or unique_run_id()
        if self.cipher is None:
            key = self.settings.encryption_key if self.settings.encrypt_state else None
            self.cipher = PayloadCipher.from_key(key)
        if self.archive is None:
            self.archive = EvidenceArchive(
                self.settings.evidence_dir,
                enabled=self.settings.archive_enabled,
                max_chars=self.settings.archive_max_chars,
            )
        if self.evidence.archive is None:
            self.evidence.archive = self.archive
        if self.llm_breaker is None:
            endpoint = self.settings.openrouter_base_url
            self.llm_breaker = get_shared_circuit_breaker(
                f"{self.settings.llm_provider}:{endpoint}",
                self.settings.circuit_failure_threshold,
                self.settings.circuit_recovery_seconds,
            )
        if self.llm_budget is None:
            self.llm_budget = CallBudget(self.settings.max_llm_calls_per_run)
        if self.dead_letters is None:
            self.dead_letters = DeadLetterQueue(
                self.settings.dead_letter_dir, run_id, cipher=self.cipher
            )
        if self.run_state is None:
            self.run_state = RunStateStore(
                self.settings.state_dir, run_id, cipher=self.cipher
            )
        if self.watchdog is None:
            self.watchdog = RunWatchdog(self.settings.max_run_seconds, self.settings.max_track_seconds)
        if self.observer is None:
            self.observer = RunObserver(
                self.settings.observability_dir,
                run_id,
                sample_rate=self.settings.trace_sample_rate,
                otel_endpoint=self.settings.otel_endpoint,
            )
        if self.http is None:
            self.http = HttpClient(
                timeout=self.settings.http_timeout,
                user_agent=self.settings.user_agent,
                cache_dir=self.settings.cache_dir,
                cache_enabled=self.settings.cache_enabled,
                cache_ttl_hours=self.settings.cache_ttl_hours,
                cache_max_bytes=self.settings.cache_max_bytes,
                cache_max_entries=self.settings.cache_max_entries,
                requests_per_minute=self.settings.requests_per_minute,
                min_seconds_between_requests=self.settings.min_seconds_between_requests,
                max_retries=self.settings.http_max_retries,
                allow_network=self.settings.allow_network,
                allow_private_network=self.settings.allow_private_network,
                max_response_bytes=self.settings.max_response_bytes,
                max_redirects=self.settings.http_max_redirects,
            )
        if self.governance is None:
            self.governance = CrawlerPolicy(self.settings, self.http)

    def close(self) -> None:
        if self.http is not None:
            self.http.close()
            self.http = None

    def __enter__(self) -> ResearchContext:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
