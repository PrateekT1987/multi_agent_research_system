"""End-to-end tests for the research pipeline.

The hermetic test runs the whole programme with the test-only deterministic
provider: plan -> three parallel research tracks -> edit -> audit -> artefacts.

The second test points the pipeline at :func:`fake_openrouter`, a local server
that speaks the OpenAI chat-completions protocol.  That exercises the real
client stack (``langchain-openrouter`` -> OpenRouter SDK -> HTTP -> CrewAI's
tool-calling loop) end to end without an API key.
"""

from __future__ import annotations

import json

from mars.config import build_settings
from mars.pipeline import ResearchPipeline


def _run(settings):
    messages: list[str] = []
    pipeline = ResearchPipeline(settings, reporter=messages.append)
    result = pipeline.run()
    return result, messages


def test_hermetic_run_produces_a_cited_report(settings) -> None:
    result, messages = _run(settings)

    report = result.report
    assert report.topic == "agentic AI"
    assert report.domain == "Agentic AI"
    # The plan is normalised onto the fixed Agentic AI taxonomy.
    assert {section.track for section in report.sections} == set(settings.tracks)
    assert all(section.findings for section in report.sections)
    assert report.audit.grounded_ratio > 0
    assert not report.audit.dangling_citations
    assert report.metrics["sources"] >= 30
    assert report.metrics["findings_recorded"] == len(result.evidence.claims())
    assert report.metrics["claims_in_report"] == report.audit.total_claims
    assert report.metrics["claims_cited"] == report.audit.cited_claims
    assert report.metrics["coverage"] == (
        f"{report.metrics['tracks_covered']}/{report.metrics['tracks_planned']}"
    )
    assert any("Mode: live LLM via OpenRouter" in message for message in messages)
    # P2: evidence is archived, and the audit is explicit about verification.
    assert report.metrics["evidence_archive"]["entries"] > 0
    assert report.audit.audit_method in {"llm+entailment", "mechanical+entailment"}
    assert report.audit.status in {"verified", "degraded", "unreviewed"}
    assert report.audit.entailment_total > 0


def test_hermetic_run_drives_the_native_tool_loop(settings) -> None:
    """The offline provider must exercise the same tool loop as a live run."""
    result, _ = _run(settings)

    counts = result.evidence.tool_call_counts()
    assert counts.get("query_corpus", 0) > 0
    assert result.report.metrics["tool_calls_by_tool"]["query_corpus"] > 0
    assert "### Tools used" in result.report_markdown


def test_hermetic_run_writes_every_artefact(settings) -> None:
    result, _ = _run(settings)

    for key in ("report_md", "report_json", "sources_json", "plan_json", "run_json"):
        path = result.output_paths[key]
        with open(path, encoding="utf-8") as handle:
            assert handle.read(), f"{key} is empty"

    markdown = result.report_markdown
    assert markdown.startswith("---")
    assert "## References" in markdown
    assert "## Methodology & Provenance" in markdown
    assert "## Grounding Audit (Appendix)" in markdown
    assert "[S1]" in markdown

    sources = json.loads(open(result.output_paths["sources_json"], encoding="utf-8").read())
    assert len(sources) == len(result.evidence.sources())
    assert {"id", "url", "title", "kind"} <= set(sources[0])

    plan = json.loads(open(result.output_paths["plan_json"], encoding="utf-8").read())
    assert plan["domain"] == "Agentic AI"
    assert {track["track"] for track in plan["tracks"]} == set(settings.tracks)


def test_pipeline_runs_against_a_live_style_endpoint(fake_openrouter, tmp_path) -> None:
    """The whole pipeline, but talking to a local OpenAI-compatible server."""
    settings = build_settings(
        topic="agentic AI",
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        openrouter_base_url=fake_openrouter,
        cache_enabled=False,
        output_dir=tmp_path / "out",
        max_rounds=1,
        env_file=None,
        allow_network=False,
        extra={"test_corpus": True},
    )

    result, _ = _run(settings)

    assert result.report.sections
    assert result.usage.calls > 0
    assert result.output_paths["report_md"]
    assert "## References" in result.report_markdown
    # The fake server asks for a corpus query, so the tool loop must have run.
    assert result.evidence.tool_call_counts().get("query_corpus", 0) > 0
    assert result.evidence.sources()


def test_track_failure_is_recorded_not_crashed(settings, monkeypatch) -> None:
    import mars.pipeline as pipeline_module

    original = pipeline_module.run_research

    def boom(*args, **kwargs):
        # One track fails; the rest of the programme must still complete.
        if len(args) > 3 and args[3] == "safety":
            raise RuntimeError("track exploded")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_research", boom)

    result, messages = _run(settings)

    assert any("failed" in message for message in messages)
    failures = [s for s in result.report.sections if s.warnings]
    assert failures
    assert "track exploded" in failures[0].warnings[0]
    # The surviving tracks still produced evidence and a report.
    assert len(result.report.sections) > len(failures)
    assert result.report.sources
