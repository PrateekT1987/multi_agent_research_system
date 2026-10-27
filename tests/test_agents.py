"""Tests for the prompt builders and agent wiring."""

from __future__ import annotations

from mars.agents.factory import plan_to_dict
from mars.agents.tasks import (
    analyst_task,
    auditor_task,
    director_task,
    editor_task,
    gap_researcher_task,
    render_sources_block,
    researcher_task,
    safe_text,
    track_title,
    writer_task,
)
from mars.schemas import ResearchPlan, Source, TrackPlan


def test_safe_text_neutralises_crewai_placeholders() -> None:
    text = 'A finding about {"json": true} and a {date} and {not_a_placeholder}.'
    safe = safe_text(text)

    assert "{json}" not in safe
    assert "{date}" not in safe
    assert "{not_a_placeholder}" not in safe
    # Braces that are not placeholder-shaped are left alone.
    assert "{" in safe


def test_track_title_is_domain_aware() -> None:
    assert track_title("foundations", "agentic AI") == "Foundations and Origins"
    assert (
        track_title("capabilities", "agentic AI")
        == "Capabilities, Benchmarks and Evaluation"
    )
    assert track_title("custom", "x") == "Custom"


def test_editor_task_lists_the_track_titles_it_was_given() -> None:
    description, _ = editor_task(
        "agentic AI",
        "Agentic AI: A Report",
        "sections",
        "findings",
        "timeline",
        ["Foundations", "Agent Architectures", "Future Trajectory"],
    )

    assert "## 1. Foundations" in description
    assert "## 3. Future Trajectory" in description
    assert "## 4. Timeline of Key Events" in description
    assert "## 5. Open Questions and Uncertainty" in description
    assert "## 6. Implications and What to Watch" in description


def test_editor_task_falls_back_to_the_canonical_structure() -> None:
    description, _ = editor_task("agentic AI", "Title", "s", "f", "t")

    # With no track_titles provided, editor uses empty headings (dynamic tracks)
    assert "## Executive Summary" in description
    assert "Timeline of Key Events" in description
    assert "Open Questions and Uncertainty" in description
    assert "Implications and What to Watch" in description


def test_writer_task_embeds_findings_and_the_heading() -> None:
    description, expected = writer_task(
        "agentic AI", "foundations", "dossier text", '{"findings": []}'
    )

    assert "## Foundations and Origins" in description
    assert '{"findings": []}' in description
    assert "CITATION RULES" in description
    assert "500-900 words" in expected


def test_analyst_and_auditor_tasks_ask_for_json() -> None:
    _, analyst_expected = analyst_task("foundations", "dossier")
    _, auditor_expected = auditor_task("draft", "registry")

    assert "Output JSON only" in analyst_expected
    assert "Output JSON only" in auditor_expected
    assert "dangling" in auditor_expected.lower()


def test_researcher_task_mentions_the_tools_it_should_use() -> None:
    description, _ = researcher_task("agentic AI", "foundations")
    for tool in ("web_search", "fetch_page", "arxiv_search", "record_finding"):
        assert tool in description
    assert "DOMAIN LOCK" in description

    gap_description, _ = researcher_task(
        "agentic AI", "foundations", gap_questions=["What about X?"]
    )
    assert "What about X?" in gap_description


def test_director_task_lists_the_tracks() -> None:
    description, expected = director_task(
        "agentic AI", ["foundations", "architectures", "capabilities"]
    )

    assert '"foundations"' in description
    assert "search_queries" in expected


def test_gap_researcher_task_lists_the_gaps() -> None:
    description, _ = gap_researcher_task("agentic AI", ["Gap one", "Gap two"])

    assert "Gap one" in description
    assert "Gap two" in description


def test_render_sources_block_lists_ids_and_truncates() -> None:
    sources = [
        Source(id=f"S{index}", url=f"https://example.com/{index}", title=f"Title {index}")
        for index in range(1, 4)
    ]
    block = render_sources_block(sources)

    assert "[S1] Title 1" in block
    assert "https://example.com/1" in block
    assert render_sources_block([]) == "NO SOURCES ARE REGISTERED YET."


def test_plan_to_dict_shapes_the_prompt_payload() -> None:
    plan = ResearchPlan(
        topic="agentic AI",
        tracks=[TrackPlan(track="foundations", objective="Establish the foundations.")],
    )

    payload = plan_to_dict(plan)

    assert payload is not None
    assert payload["track"]["track"] == "foundations"
    assert payload["tracks"][0]["objective"] == "Establish the foundations."
    assert plan_to_dict(None) is None
