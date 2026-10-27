"""The evidence store: a thread-safe registry of sources and claims.

Every tool that touches the outside world registers what it found here.  Agents
cite sources by short id (``[S3]``) and the store is the single source of truth
used by the grounding auditor, the report renderer, and the metrics collector.

The store is safe for concurrent use because tracks are researched in parallel.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from urllib.parse import urlsplit, urlunsplit

from .archive import EvidenceArchive
from .schemas import Claim, Source

__all__ = ["EvidenceStore", "normalise_url"]

_CITATION_RE = re.compile(r"\[(S\d+)\]")
# Shortest block we are willing to treat as a prose paragraph.  Anything
# shorter (list items, table rows, code fences, stray headings) is ignored by
# the grounding heuristics.
_MIN_PROSE_WORDS = 12
_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "ref_src",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
}


def _source_number(source_id: str) -> int:
    try:
        return int(str(source_id).lstrip("S") or 0)
    except ValueError:
        return 0


def normalise_url(url: str) -> str:
    """Canonicalise a URL so the same page is not registered twice."""
    if not url:
        return ""
    url = url.strip()
    if url.startswith("//"):
        url = "https:" + url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.scheme:
        parts = urlsplit("https://" + url)
    query = "&".join(
        piece
        for piece in parts.query.split("&")
        if piece and piece.split("=", 1)[0] not in _TRACKING_PARAMS
    )
    path = parts.path.rstrip("/") or "/"
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


def _host(url: str) -> str:
    """Return the lower-case host of a URL (``""`` when it has none)."""
    try:
        parts = urlsplit(normalise_url(url))
    except ValueError:
        return ""
    return (parts.hostname or "").lower()


class EvidenceStore:
    """Registry of sources plus the claims that depend on them."""

    def __init__(self, archive: EvidenceArchive | None = None) -> None:
        self._lock = threading.RLock()
        self._sources: dict[str, Source] = {}
        self._by_url: dict[str, str] = {}
        self._claims: dict[str, Claim] = {}
        self._next_source = 1
        self._next_claim = 1
        self._tool_calls: dict[str, int] = {}
        self.archive = archive

    # ------------------------------------------------------------------ URLs
    def register(
        self,
        url: str,
        title: str = "",
        snippet: str = "",
        *,
        kind: str = "unknown",
        provider: str = "unknown",
        published: str | None = None,
        content: str | None = None,
        content_type: str = "",
        http_status: int = 0,
        final_url: str = "",
        quality: float = 0.0,
        license: str = "unknown",
        paywalled: bool = False,
        injection_flags: Iterable[str] = (),
    ) -> Source:
        """Register a source (idempotent on the normalised URL).

        When ``content`` is supplied it is written to the immutable evidence
        archive and the resulting content hash is recorded on the source.
        """
        key = normalise_url(url)
        with self._lock:
            existing_id = self._by_url.get(key)
            if existing_id:
                source = self._sources[existing_id]
                # Enrich an existing record when a later fetch has better data.
                if snippet and len(snippet) > len(source.snippet):
                    source.snippet = snippet
                if published and not source.published:
                    source.published = published
                if quality and quality > source.quality:
                    source.quality = quality
                if license and license != "unknown":
                    source.license = license
                source.paywalled = source.paywalled or paywalled
                for flag in injection_flags:
                    if flag not in source.injection_flags:
                        source.injection_flags.append(flag)
                if content and not source.content_hash:
                    self._attach_archive(
                        source,
                        content,
                        title=title,
                        content_type=content_type,
                        http_status=http_status,
                        final_url=final_url,
                        provider=provider,
                        injection_flags=injection_flags,
                    )
                return source

            source_id = f"S{self._next_source}"
            self._next_source += 1
            source = Source(
                id=source_id,
                url=url,
                title=title or url,
                snippet=snippet,
                kind=kind,  # type: ignore[arg-type]
                provider=provider,
                published=published,
                quality=quality,
                license=license,
                paywalled=paywalled,
                injection_flags=list(dict.fromkeys(injection_flags)),
            )
            if content:
                self._attach_archive(
                    source,
                    content,
                    title=title,
                    content_type=content_type,
                    http_status=http_status,
                    final_url=final_url,
                    provider=provider,
                    injection_flags=injection_flags,
                )
            self._sources[source_id] = source
            self._by_url[key] = source_id
            return source

    def _attach_archive(
        self,
        source: Source,
        content: str,
        *,
        title: str,
        content_type: str,
        http_status: int,
        final_url: str,
        provider: str,
        injection_flags: Iterable[str],
    ) -> None:
        if self.archive is None:
            return
        record = self.archive.store(
            content,
            url=source.url,
            title=title or source.title,
            content_type=content_type,
            http_status=http_status,
            final_url=final_url,
            provider=provider,
            retrieved_at=source.retrieved_at,
            injection_flags=tuple(injection_flags),
        )
        if record is not None:
            source.content_hash = record.content_hash
            source.archive_path = record.path
            source.final_url = record.final_url
            source.content_type = record.content_type
            source.http_status = record.http_status

    def get(self, source_id: str) -> Source | None:
        with self._lock:
            return self._sources.get(source_id.upper())

    def restore(self, sources: list[Source]) -> None:
        """Rehydrate the registry from a persisted snapshot, preserving ids."""
        with self._lock:
            for source in sorted(sources, key=lambda item: _source_number(item.id)):
                self._sources[source.id] = source
                self._by_url[normalise_url(source.url)] = source.id
                number = _source_number(source.id)
                if number >= self._next_source:
                    self._next_source = number + 1

    def find_by_url(self, url: str) -> Source | None:
        key = normalise_url(url)
        with self._lock:
            source_id = self._by_url.get(key)
            return self._sources.get(source_id) if source_id else None

    def host_source_count(self, url: str) -> int:
        """How many registered sources share this URL's host."""
        host = _host(url)
        if not host:
            return 0
        with self._lock:
            return sum(1 for source in self._sources.values() if _host(source.url) == host)

    def domain_cap_reached(self, url: str, cap: int | None) -> bool:
        """Whether registering ``url`` would exceed the per-domain cap.

        The cap is advisory for the retrieval tools rather than a property of
        the registry itself: the same URL is always idempotent, and a cap of
        zero (or ``None``) disables the check.
        """
        if not cap or cap <= 0:
            return False
        if self.find_by_url(url) is not None:
            return False
        return self.host_source_count(url) >= int(cap)

    def flagged_source_ids(self) -> set[str]:
        """Source ids that carry at least one prompt-injection flag."""
        with self._lock:
            return {
                source_id
                for source_id, source in self._sources.items()
                if source.injection_flags
            }

    # ---------------------------------------------------------------- claims
    def add_claim(
        self,
        text: str,
        evidence: Iterable[str] = (),
        *,
        confidence: float = 0.5,
        track: str = "general",
        kind: str = "fact",
    ) -> Claim:
        """Record a claim and the source ids that support it."""
        cleaned = [e.strip().upper() for e in evidence if e and e.strip()]
        with self._lock:
            claim_id = f"C{self._next_claim}"
            self._next_claim += 1
            claim = Claim(
                id=claim_id,
                text=text.strip(),
                evidence=cleaned,
                confidence=confidence,
                track=track,
                kind=kind,  # type: ignore[arg-type]
            )
            self._claims[claim_id] = claim
            return claim

    def claims(self, track: str | None = None) -> list[Claim]:
        with self._lock:
            values = list(self._claims.values())
        if track:
            values = [c for c in values if c.track == track]
        return values

    def sources(self) -> list[Source]:
        with self._lock:
            return list(self._sources.values())

    def source_ids(self) -> list[str]:
        with self._lock:
            return list(self._sources.keys())

    # ------------------------------------------------------------- utilities
    def record_tool_call(self, tool_name: str) -> None:
        with self._lock:
            self._tool_calls[tool_name] = self._tool_calls.get(tool_name, 0) + 1

    def tool_call_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._tool_calls)

    @staticmethod
    def extract_citations(text: str) -> list[str]:
        """Return the citation ids referenced in ``text`` (order preserved)."""
        seen: list[str] = []
        for match in _CITATION_RE.findall(text or ""):
            cid = match.upper()
            if cid not in seen:
                seen.append(cid)
        return seen

    def dangling_citations(self, text: str) -> list[str]:
        """Citations in ``text`` that point at an unregistered source."""
        with self._lock:
            known = set(self._sources)
        return [cid for cid in self.extract_citations(text) if cid not in known]

    @staticmethod
    def _prose_blocks(text: str, min_words: int) -> list[str]:
        """Split ``text`` into prose paragraphs, ignoring structural blocks."""
        blocks: list[str] = []
        for block in (text or "").split("\n\n"):
            stripped = block.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("|") or stripped.startswith("```"):
                continue
            if len(stripped.split()) < min_words:
                continue
            blocks.append(" ".join(stripped.split()))
        return blocks

    def uncited_paragraphs(self, text: str, *, min_words: int = _MIN_PROSE_WORDS) -> list[str]:
        """Paragraphs that make an assertion but carry no citation."""
        return [
            block[:200]
            for block in self._prose_blocks(text, min_words)
            if not _CITATION_RE.search(block)
        ]

    def grounding_score(self, text: str) -> tuple[float, int, int]:
        """Compute a simple grounding ratio for a body of text.

        Returns ``(ratio, cited_blocks, total_blocks)`` where blocks are
        paragraphs of prose (headings, tables and code fences are ignored).
        """
        blocks = self._prose_blocks(text, _MIN_PROSE_WORDS)
        total = len(blocks)
        cited = sum(1 for block in blocks if _CITATION_RE.search(block))
        if total == 0:
            return 1.0, 0, 0
        return cited / total, cited, total

    def unused_sources(self, text: str) -> list[Source]:
        cited = set(self.extract_citations(text))
        return [s for s in self.sources() if s.id not in cited]

    def render_reference_list(self, source_ids: Iterable[str] | None = None) -> str:
        """Render a Markdown reference list for the given (or all) sources."""
        wanted = [s.upper() for s in source_ids] if source_ids else None
        lines: list[str] = []
        for source in self.sources():
            if wanted and source.id not in wanted:
                continue
            published = f" — {source.published}" if source.published else ""
            lines.append(
                f"- **[{source.id}]** {source.title}. "
                f"<{source.url}> ({source.kind}, {source.provider}){published}"
            )
        return "\n".join(lines)

    def summary(self) -> dict[str, object]:
        with self._lock:
            kinds: dict[str, int] = {}
            providers: dict[str, int] = {}
            for source in self._sources.values():
                kinds[source.kind] = kinds.get(source.kind, 0) + 1
                providers[source.provider] = providers.get(source.provider, 0) + 1
            verification: dict[str, int] = {}
            for claim in self._claims.values():
                verification[claim.verification] = verification.get(claim.verification, 0) + 1
            archived = sum(1 for source in self._sources.values() if source.content_hash)
            flagged = sum(1 for source in self._sources.values() if source.injection_flags)
            return {
                "sources": len(self._sources),
                "archived_sources": archived,
                "sources_with_injection_flags": flagged,
                "ledger_claims": len(self._claims),
                "claim_verification": verification,
                "tool_calls": sum(self._tool_calls.values()),
                "sources_by_kind": kinds,
                "sources_by_provider": providers,
                "tool_calls_by_tool": dict(self._tool_calls),
                "claims_without_evidence": sum(1 for claim in self._claims.values() if not claim.evidence),
                "claims_with_unknown_evidence": sum(
                    1
                    for claim in self._claims.values()
                    if any(source_id not in self._sources for source_id in claim.evidence)
                ),
            }

    def verify_claims(self) -> dict[str, object]:
        """Run the deterministic entailment audit over the claim ledger."""
        from .entailment import audit_claims

        report = audit_claims(self, self.claims())
        return report.as_dict()

    def entailment_audit(self, text: str, *, supported_at: float = 0.55, partial_at: float = 0.30):
        """Run the deterministic entailment audit over a body of text."""
        from .entailment import audit_text

        return audit_text(self, text, supported_at=supported_at, partial_at=partial_at)
