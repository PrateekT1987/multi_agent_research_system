"""P2: crawler governance (robots.txt, domains, quality, paywall, licensing)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mars.config import build_settings
from mars.governance import (
    CrawlerPolicy,
    detect_license,
    detect_paywall,
    filter_search_results,
    parse_robots,
    source_quality,
)
from mars.net import Response
from mars.tools.search import SearchResult


def _settings(**overrides):
    base = {
        "llm_provider": "openrouter",
        "openrouter_api_key": "sk-test",
        "allow_private_network": True,
        "env_file": None,
    }
    base.update(overrides)
    return build_settings(**base)


class _StubHttp:
    def __init__(self, robots: str = "") -> None:
        self.robots = robots
        self.calls: list[str] = []

    def get(self, url: str, **_kwargs) -> Response:
        self.calls.append(url)
        return Response(
            url=url,
            status=200,
            text=self.robots,
            headers={"content-type": "text/plain"},
        )


def test_parse_robots_builds_user_agent_groups() -> None:
    groups = parse_robots(
        "User-agent: *\nDisallow: /private\nAllow: /private/public\n\n"
        "User-agent: MarsResearchBot\nDisallow: /\n"
    )

    assert len(groups) == 2
    assert groups[0].agents == ["*"]
    assert ("disallow", "/private") in groups[0].rules
    assert groups[1].agents == ["marsresearchbot"]


def test_source_quality_ranks_official_above_social() -> None:
    assert source_quality("https://www.nih.gov/report") > source_quality(
        "https://twitter.com/thread"
    )
    assert source_quality("https://arxiv.org/abs/1234") > source_quality(
        "https://example.com/blog/post"
    )


def test_paywall_and_license_detection() -> None:
    assert detect_paywall("Subscribe to continue reading this article.")
    assert not detect_paywall("A freely available article.")
    assert detect_license("Released under the MIT License.") == "MIT"
    assert detect_license("This work is in the public domain.") == "public-domain"
    assert detect_license("Nothing here.") == "unknown"


def test_blocked_domain_is_refused() -> None:
    policy = CrawlerPolicy(_settings(blocked_domains=("example.com",)))

    decision = policy.check_url("https://example.com/a")

    assert decision.allowed is False
    assert "blocklist" in decision.reason


def test_allowlist_refuses_other_domains() -> None:
    policy = CrawlerPolicy(_settings(allowed_domains=("arxiv.org",)))

    assert policy.check_url("https://arxiv.org/abs/1").allowed is True
    assert policy.check_url("https://example.com/a").allowed is False


def test_blocked_url_pattern_is_refused() -> None:
    policy = CrawlerPolicy(_settings(blocked_url_patterns=(r"/login",)))

    assert policy.check_url("https://example.com/login").allowed is False
    assert policy.check_url("https://example.com/article").allowed is True


def test_minimum_source_quality_is_enforced() -> None:
    policy = CrawlerPolicy(_settings(min_source_quality=0.8))

    assert policy.check_url("https://twitter.com/x").allowed is False
    assert policy.check_url("https://arxiv.org/abs/1").allowed is True


def test_stale_sources_are_refused() -> None:
    old = (datetime.now(UTC) - timedelta(days=400)).strftime("%Y-%m-%d")
    fresh = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
    policy = CrawlerPolicy(_settings(max_source_age_days=30))

    assert policy.check_url("https://arxiv.org/abs/1", published=old).allowed is False
    assert policy.check_url("https://arxiv.org/abs/1", published=fresh).allowed is True


def test_paywalled_content_is_refused_when_configured() -> None:
    policy = CrawlerPolicy(_settings(block_paywalled=True))

    decision = policy.check_content(
        "https://example.com/a", "Subscribe to continue reading this article."
    )

    assert decision.allowed is False
    assert decision.paywalled is True
    assert "paywalled" in decision.reason


def test_robots_disallow_blocks_a_fetch() -> None:
    http = _StubHttp("User-agent: *\nDisallow: /private\n")
    policy = CrawlerPolicy(_settings(respect_robots=True), http)

    decision = policy.check_fetch("https://example.com/private/page")

    assert decision.allowed is False
    assert decision.robots_allowed is False
    assert "robots.txt" in decision.reason
    assert http.calls and http.calls[0].endswith("/robots.txt")


def test_robots_allow_permits_a_fetch() -> None:
    http = _StubHttp("User-agent: *\nDisallow: /private\n")
    policy = CrawlerPolicy(_settings(respect_robots=True), http)

    assert policy.check_fetch("https://example.com/public").allowed is True


def test_robots_can_be_disabled() -> None:
    http = _StubHttp("User-agent: *\nDisallow: /\n")
    policy = CrawlerPolicy(_settings(respect_robots=False), http)

    assert policy.check_fetch("https://example.com/private").allowed is True
    assert http.calls == []


def test_filter_search_results_partitions_and_annotates() -> None:
    settings = _settings(blocked_domains=("spam.example",))
    results = [
        SearchResult(
            title="Good",
            url="https://arxiv.org/abs/1",
            snippet="LLM agents use tools to plan and act.",
        ),
        SearchResult(
            title="Bad",
            url="https://spam.example/x",
            snippet="LLM agents use tools to plan and act.",
        ),
    ]

    accepted, rejected = filter_search_results(results, settings)

    assert [item.title for item in accepted] == ["Good"]
    assert rejected and rejected[0][0].title == "Bad"
    assert accepted[0].quality > 0
    assert accepted[0].license == "unknown"
    assert accepted[0].paywalled is False
