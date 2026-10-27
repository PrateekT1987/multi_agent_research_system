"""Crawler governance: robots.txt, domain policy, quality and licensing.

Mars is a polite crawler.  Every fetch goes through this policy, and search
results are validated before they are registered as citable evidence.  The
policy is deliberately configurable: an operator can restrict a run to an
allowlist, block a domain, refuse paywalled pages, or reject stale sources.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

from .domain import DOMAIN, is_domain_relevant
from .net import HttpClient, HttpError, validate_url

__all__ = [
    "CrawlerPolicy",
    "GovernanceDecision",
    "RobotsCache",
    "filter_search_results",
    "source_quality",
    "detect_paywall",
    "detect_license",
    "parse_robots",
]

_HIGH_QUALITY_SUFFIXES = (".gov", ".edu", ".mil", ".int")
_HIGH_QUALITY_DOMAINS = {
    "arxiv.org",
    "acm.org",
    "ieee.org",
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "nih.gov",
    "who.int",
    "un.org",
    "oecd.org",
    "worldbank.org",
    "europa.eu",
    "wikipedia.org",
    "github.com",
    "openai.com",
    "anthropic.com",
    "deepmind.google",
    "ai.googleblog.com",
    "research.google",
    "microsoft.com",
    "aws.amazon.com",
    "cloud.google.com",
}
_LOW_QUALITY_DOMAINS = {
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "reddit.com",
    "quora.com",
    "medium.com",
    "substack.com",
    "blogspot.com",
    "wordpress.com",
    "pinterest.com",
}
_PAYWALL_MARKERS = (
    "subscribe to continue",
    "subscribe to read",
    "sign in to read",
    "sign in to continue",
    "this article is for subscribers",
    "subscriber-only",
    "members-only",
    "you have reached your free article limit",
    "create a free account to continue",
    "purchase this article",
)
_LICENSE_PATTERNS = (
    (re.compile(r"creative commons(?:\s+attribution)?(?:\s+([0-9.]+))?", re.I), "CC"),
    (re.compile(r"\bCC[ -]BY(?:[ -][A-Z]{2})?(?:[ -][0-9.]+)?\b", re.I), "CC-BY"),
    (re.compile(r"\bMIT License\b", re.I), "MIT"),
    (re.compile(r"\bApache License\b", re.I), "Apache-2.0"),
    (re.compile(r"\bpublic domain\b", re.I), "public-domain"),
    (re.compile(r"\ball rights reserved\b", re.I), "all-rights-reserved"),
)


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return ""


def _domain_matches(host: str, candidate: str) -> bool:
    candidate = candidate.strip().lower().lstrip(".")
    if not candidate:
        return False
    return host == candidate or host.endswith("." + candidate)


def source_quality(url: str) -> float:
    """A coarse 0-1 reputation score used as a configurable filter.

    This is intentionally simple and explainable: official/peer-reviewed
    domains rank above general web pages, which rank above social/UGC.
    """
    host = _host(url)
    if not host:
        return 0.0
    if any(host.endswith(suffix) for suffix in _HIGH_QUALITY_SUFFIXES):
        return 0.95
    if any(_domain_matches(host, domain) for domain in _HIGH_QUALITY_DOMAINS):
        return 0.85
    if any(_domain_matches(host, domain) for domain in _LOW_QUALITY_DOMAINS):
        return 0.25
    if host.startswith("blog.") or host.startswith("docs."):
        return 0.65
    return 0.5


def detect_paywall(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _PAYWALL_MARKERS)


def detect_license(text: str) -> str:
    for pattern, label in _LICENSE_PATTERNS:
        if pattern.search(text or ""):
            return label
    return "unknown"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(raw[: len(datetime.now().strftime(fmt))], fmt)
            return parsed.replace(tzinfo=UTC)
        except ValueError:
            continue
    match = re.search(r"\b(19|20)\d{2}\b", raw)
    if match:
        return datetime(int(match.group(0)), 1, 1, tzinfo=UTC)
    return None


@dataclass(slots=True)
class GovernanceDecision:
    """The outcome of validating one URL (and optionally its content)."""

    allowed: bool
    reason: str = ""
    quality: float = 0.0
    paywalled: bool = False
    license: str = "unknown"
    robots_allowed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "quality": round(self.quality, 3),
            "paywalled": self.paywalled,
            "license": self.license,
            "robots_allowed": self.robots_allowed,
        }


@dataclass(slots=True)
class _RobotsGroup:
    agents: list[str] = field(default_factory=list)
    rules: list[tuple[str, str]] = field(default_factory=list)  # (allow|disallow, path)


def parse_robots(text: str) -> list[_RobotsGroup]:
    """Parse a robots.txt document into user-agent groups."""
    groups: list[_RobotsGroup] = []
    current: _RobotsGroup | None = None
    expecting_agent = False
    for raw_line in (text or "").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "user-agent":
            if current is None or not expecting_agent:
                current = _RobotsGroup()
                groups.append(current)
            current.agents.append(value.lower())
            expecting_agent = True
            continue
        if key in {"allow", "disallow"}:
            if current is None:
                current = _RobotsGroup(agents=["*"])
                groups.append(current)
            current.rules.append((key, value))
            expecting_agent = False
    return groups


def _robots_allows(text: str, path: str, user_agent: str) -> bool:
    groups = parse_robots(text)
    agent = (user_agent or "").lower()
    specific: list[_RobotsGroup] = []
    wildcard: list[_RobotsGroup] = []
    for group in groups:
        if any(token == "*" for token in group.agents):
            wildcard.append(group)
        if any(token and token in agent for token in group.agents if token != "*"):
            specific.append(group)
    selected = specific or wildcard
    if not selected:
        return True

    best: tuple[int, str] | None = None
    for group in selected:
        for directive, rule in group.rules:
            if not rule:
                continue
            if not path.startswith(rule):
                continue
            candidate = (len(rule), directive)
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0] and directive == "allow"
            ):
                best = candidate
    if best is None:
        return True
    return best[1] == "allow"


class RobotsCache:
    """Thread-safe, per-host robots.txt cache."""

    def __init__(self, http: HttpClient | None, user_agent: str, *, fail_closed: bool = False) -> None:
        self.http = http
        self.user_agent = user_agent
        self.fail_closed = fail_closed
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()

    def _document(self, url: str) -> str | None:
        host = _host(url)
        if not host:
            return None
        with self._lock:
            if host in self._cache:
                return self._cache[host]
        if self.http is None:
            return None
        parts = urlsplit(url)
        robots_url = f"{parts.scheme or 'https'}://{parts.netloc}/robots.txt"
        try:
            response = self.http.get(
                robots_url,
                use_cache=True,
                allowed_content_types={"text/plain", "text/html", "application/xml"},
            )
        except HttpError:
            with self._lock:
                self._cache[host] = ""
            return "" if not self.fail_closed else None
        if response.status >= 400:
            document = ""
        else:
            document = response.text
        with self._lock:
            self._cache[host] = document
        return document

    def allowed(self, url: str) -> bool:
        document = self._document(url)
        if document is None:
            return not self.fail_closed
        if not document.strip():
            return True
        parts = urlsplit(url)
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        return _robots_allows(document, path, self.user_agent)


class CrawlerPolicy:
    """Validate destinations and fetched content against the run's policy."""

    def __init__(self, settings: Any, http: HttpClient | None = None) -> None:
        self.settings = settings
        self.http = http
        self.robots = RobotsCache(
            http,
            getattr(settings, "user_agent", "MarsResearchBot/0.1"),
            fail_closed=bool(getattr(settings, "robots_fail_closed", False)),
        )

    def _domain_allowed(self, host: str) -> tuple[bool, str]:
        allowed = tuple(getattr(self.settings, "allowed_domains", ()) or ())
        blocked = tuple(getattr(self.settings, "blocked_domains", ()) or ())
        if any(_domain_matches(host, domain) for domain in blocked):
            return False, f"domain {host!r} is on the blocklist"
        if allowed and not any(_domain_matches(host, domain) for domain in allowed):
            return False, f"domain {host!r} is not on the allowlist"
        return True, ""

    def _pattern_allowed(self, url: str) -> tuple[bool, str]:
        for pattern in getattr(self.settings, "blocked_url_patterns", ()) or ():
            try:
                if re.search(pattern, url, re.I):
                    return False, f"URL matches blocked pattern {pattern!r}"
            except re.error:
                continue
        return True, ""

    def check_url(
        self,
        url: str,
        *,
        published: str | None = None,
        text: str | None = None,
        check_robots: bool = False,
    ) -> GovernanceDecision:
        """Validate one URL, its recency and (when available) its content."""
        try:
            validate_url(
                url,
                allow_private_network=bool(getattr(self.settings, "allow_private_network", False)),
            )
        except HttpError as exc:
            return GovernanceDecision(allowed=False, reason=str(exc))

        host = _host(url)
        ok, reason = self._domain_allowed(host)
        if not ok:
            return GovernanceDecision(allowed=False, reason=reason, quality=source_quality(url))
        ok, reason = self._pattern_allowed(url)
        if not ok:
            return GovernanceDecision(allowed=False, reason=reason, quality=source_quality(url))

        quality = source_quality(url)
        minimum = float(getattr(self.settings, "min_source_quality", 0.0) or 0.0)
        if quality < minimum:
            return GovernanceDecision(
                allowed=False,
                reason=f"source quality {quality:.2f} is below the minimum {minimum:.2f}",
                quality=quality,
            )

        max_age = getattr(self.settings, "max_source_age_days", None)
        if max_age is not None and published:
            parsed = _parse_date(published)
            if parsed is not None:
                cutoff = datetime.now(UTC) - timedelta(days=max(0, int(max_age)))
                if parsed < cutoff:
                    return GovernanceDecision(
                        allowed=False,
                        reason=f"source is older than {max_age} day(s)",
                        quality=quality,
                    )

        paywalled = detect_paywall(text or "")
        license_name = detect_license(text or "")
        if paywalled and bool(getattr(self.settings, "block_paywalled", False)):
            return GovernanceDecision(
                allowed=False,
                reason="paywalled content is disabled by policy",
                quality=quality,
                paywalled=True,
                license=license_name,
            )

        robots_allowed = True
        if check_robots and bool(getattr(self.settings, "respect_robots", True)):
            robots_allowed = self.robots.allowed(url)
            if not robots_allowed:
                return GovernanceDecision(
                    allowed=False,
                    reason="robots.txt disallows this path",
                    quality=quality,
                    paywalled=paywalled,
                    license=license_name,
                    robots_allowed=False,
                )

        return GovernanceDecision(
            allowed=True,
            quality=quality,
            paywalled=paywalled,
            license=license_name,
            robots_allowed=robots_allowed,
        )

    def check_fetch(self, url: str) -> GovernanceDecision:
        return self.check_url(url, check_robots=True)

    def check_content(self, url: str, text: str, *, published: str | None = None) -> GovernanceDecision:
        decision = self.check_url(url, published=published, text=text, check_robots=False)
        if not decision.allowed:
            return decision
        if not is_domain_relevant(text):
            return GovernanceDecision(
                allowed=False,
                reason=f"content is outside the {DOMAIN} domain",
                quality=decision.quality,
                paywalled=decision.paywalled,
                license=decision.license,
                robots_allowed=decision.robots_allowed,
            )
        return decision


def filter_search_results(
    results: list[Any],
    settings: Any,
    *,
    http: HttpClient | None = None,
) -> tuple[list[Any], list[tuple[Any, str]]]:
    """Partition search hits into accepted and rejected (with a reason)."""
    policy = CrawlerPolicy(settings, http)
    accepted: list[Any] = []
    rejected: list[tuple[Any, str]] = []
    for result in results:
        snippet = getattr(result, "snippet", "") or ""
        title = getattr(result, "title", "") or ""
        if not is_domain_relevant(f"{title} {snippet}"):
            rejected.append((result, f"result is outside the {DOMAIN} domain"))
            continue
        decision = policy.check_url(
            getattr(result, "url", ""),
            published=getattr(result, "published", None),
            text=snippet,
        )
        if decision.allowed:
            result.quality = decision.quality
            result.license = decision.license
            result.paywalled = decision.paywalled
            accepted.append(result)
        else:
            rejected.append((result, decision.reason))
    return accepted, rejected
