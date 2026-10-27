"""Tests for the Agentic AI domain lock.

Mars is not a general-purpose research system.  These tests pin the four
enforcement layers: the topic gate, the canonical track taxonomy, the prompt
protocol and retrieval filtering.
"""

from __future__ import annotations

import pytest

from mars.config import ConfigError, build_settings
from mars.domain import (
    DEFAULT_DOMAIN_TRACKS,
    DOMAIN,
    DOMAIN_TRACK_KEYS,
    DomainError,
    canonical_track,
    domain_protocol,
    ensure_domain_topic,
    is_domain_relevant,
    is_domain_topic,
    scope_query,
    track_menu,
)
from mars.governance import CrawlerPolicy, filter_search_results
from mars.pipeline import ResearchPipeline
from mars.schemas import ResearchPlan, TrackPlan
from mars.tools.search import SearchResult


def test_domain_constant_is_agentic_ai() -> None:
    assert DOMAIN == "Agentic AI"


@pytest.mark.parametrize(
    "topic",
    [
        "agentic AI",
        "agent memory",
        "multi-agent orchestration",
        "tool use and function calling",
        "computer-use agents",
        "LLM agent evaluation",
        "ReAct",
        "prompt injection in tool-using agents",
        "AI customer service agents",
    ],
)
def test_on_domain_topics_are_accepted(topic: str) -> None:
    assert is_domain_topic(topic) is True
    assert ensure_domain_topic(topic) == " ".join(topic.split())


@pytest.mark.parametrize(
    "topic",
    [
        "quantum computing",
        "CRISPR gene editing",
        "the Roman empire",
        "travel agent",
        "chemical agents",
        "impact of ongoing war between Iran and USA",
        "real estate agent",
    ],
)
def test_off_domain_topics_are_rejected(topic: str) -> None:
    assert is_domain_topic(topic) is False
    with pytest.raises(DomainError):
        ensure_domain_topic(topic)


def test_empty_topic_defaults_to_the_domain() -> None:
    assert ensure_domain_topic("   ") == DOMAIN


def test_relevance_filter_distinguishes_agents_from_other_agents() -> None:
    assert is_domain_relevant("LLM agents call tools and plan multi-step tasks")
    assert is_domain_relevant("A benchmark for computer-use agents")
    assert not is_domain_relevant("A recipe for tomato pasta")
    assert not is_domain_relevant("Travel agent reviews for summer holidays")


def test_scope_query_adds_domain_context_only_when_needed() -> None:
    assert scope_query("agent memory") == "agent memory"
    assert scope_query("prompt injection") == "prompt injection agentic AI"
    assert scope_query("") == "agentic AI"


def test_canonical_taxonomy_maps_loose_names() -> None:
    assert canonical_track("history").key == "foundations"
    assert canonical_track("current").key == "capabilities"
    assert canonical_track("multi-agent").key == "architectures"
    assert canonical_track("ethics").key == "safety"
    assert canonical_track("applications").key == "applications"
    assert canonical_track("something invented") is None


def test_prompt_protocol_and_track_menu_name_the_domain() -> None:
    protocol = domain_protocol()
    assert DOMAIN in protocol
    assert "DOMAIN LOCK" in protocol

    menu = track_menu()
    for key in DOMAIN_TRACK_KEYS:
        assert key in menu
    assert set(DEFAULT_DOMAIN_TRACKS) <= set(DOMAIN_TRACK_KEYS)


def test_settings_validate_rejects_off_domain_topic_and_unknown_tracks() -> None:
    settings = build_settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        env_file=None,
    )
    with pytest.raises(ConfigError, match="outside that domain"):
        settings.with_overrides(topic="quantum computing").validate()

    with pytest.raises(ConfigError, match="unknown track"):
        settings.with_overrides(tracks=("foundations", "astrology")).validate()


def test_settings_serialise_the_fixed_domain() -> None:
    settings = build_settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        env_file=None,
    )
    assert settings.to_dict()["domain"] == DOMAIN


def test_pipeline_normalises_onto_the_canonical_taxonomy(tmp_path) -> None:
    settings = build_settings(
        topic="agentic AI",
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        output_dir=tmp_path,
        env_file=None,
    )
    pipeline = ResearchPipeline(settings)
    plan = ResearchPlan(
        topic="agentic AI",
        tracks=[
            TrackPlan(track="history", objective="Origins."),
            TrackPlan(track="invented_track", objective="Not real."),
            TrackPlan(track="applications", objective="Deployment."),
            TrackPlan(track="history", objective="Duplicate."),
        ],
    )

    normalised = pipeline._normalise_plan(plan)

    assert [track.track for track in normalised.tracks] == [
        "foundations",
        "applications",
    ]
    assert normalised.domain == DOMAIN


def test_retrieval_filter_rejects_off_domain_results(tmp_path) -> None:
    settings = build_settings(
        topic="agentic AI",
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        output_dir=tmp_path,
        env_file=None,
    )
    results = [
        SearchResult(
            title="Agent memory architectures",
            url="https://example.com/agents",
            snippet="LLM agents use memory and tools to plan.",
        ),
        SearchResult(
            title="Best pasta recipes",
            url="https://example.com/pasta",
            snippet="Boil water and add salt.",
        ),
    ]

    accepted, rejected = filter_search_results(results, settings)

    assert [item.title for item in accepted] == ["Agent memory architectures"]
    assert rejected and rejected[0][0].title == "Best pasta recipes"
    assert "domain" in rejected[0][1]


def test_fetched_content_is_rejected_when_off_domain(tmp_path) -> None:
    settings = build_settings(
        topic="agentic AI",
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        output_dir=tmp_path,
        env_file=None,
    )
    policy = CrawlerPolicy(settings)

    assert policy.check_content("https://example.com/a", "Agents plan and use tools.").allowed
    decision = policy.check_content("https://example.com/b", "A guide to baking sourdough.")
    assert decision.allowed is False
    assert "domain" in decision.reason
