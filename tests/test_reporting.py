"""Tests for report rendering."""

from __future__ import annotations

from mars.evidence import EvidenceStore
from mars.reporting import render_front_matter, render_report
from mars.schemas import AuditResult, ResearchReport, Source, TrackResult


def _report() -> tuple[ResearchReport, EvidenceStore]:
    evidence = EvidenceStore()
    evidence.register("https://example.com/one", "First source", "snippet one")
    evidence.register("https://example.com/two", "Second source", "snippet two")

    report = ResearchReport(
        topic="agentic AI",
        title="Agentic AI: History, Current Practice, and What Comes Next",
        draft_markdown=(
            "# Research Report\n\n"
            "## Executive Summary\n\n"
            "Agents moved from symbolic control to model-driven tool use. [S1]\n\n"
            "## Origins\n\n"
            "The field began with expert systems. [S1]"
        ),
        sections=[
            TrackResult(
                track="history",
                title="Origins",
                section_markdown="## Origins\n\nThe field began with expert systems. [S1]",
                findings=[],
            )
        ],
        sources=evidence.sources(),
        audit=AuditResult(
            grounded_ratio=0.9,
            cited_claims=2,
            total_claims=2,
            verdict="pass",
            notes="Looks solid.",
        ),
        metrics={
            "grounding_threshold": 0.85,
            "sources": 2,
            "findings_recorded": 1,
            "claims_in_report": 2,
            "claims_cited": 2,
            "coverage": "1/1",
            "tracks_covered": 1,
            "tracks_planned": 1,
        },
        rounds=1,
    )
    return report, evidence


def test_front_matter_contains_run_metadata() -> None:
    report, _ = _report()
    front = render_front_matter(report)

    assert front.startswith("---\n")
    assert 'domain: "Agentic AI"' in front
    assert "grounding_score: 0.900" in front
    assert "entailment_score: 0.000" in front
    assert "audit_status: unreviewed" in front
    assert "audit_verdict: pass" in front
    assert "sources: 2" in front
    assert "findings_recorded: 1" in front
    assert "claims_in_report: 2" in front
    assert "claims_cited: 2" in front
    assert 'coverage: "1/1"' in front
    assert "\nclaims:" not in front
    assert front.rstrip().endswith("---")


def test_front_matter_reports_a_verified_audit() -> None:
    report, _ = _report()
    report.audit.verified = True
    report.audit.entailment_ratio = 0.8

    front = render_front_matter(report)

    assert "audit_status: verified" in front
    assert "entailment_score: 0.800" in front


def test_audit_appendix_explains_degradation() -> None:
    report, evidence = _report()
    report.audit.degraded = True
    report.audit.degraded_reasons = ["entailment 0.20 below threshold 0.55"]
    report.audit.entailment_ratio = 0.2
    report.audit.entailment_total = 5
    report.audit.entailment_unsupported = 2
    report.audit.audit_method = "llm+entailment"
    report.metrics["evidence_archive"] = {"entries": 4, "bytes": 1234}

    markdown = render_report(report, evidence)

    assert "Status: **degraded**" in markdown
    assert "entailment 0.20 below threshold 0.55" in markdown
    assert "Audit method: llm+entailment" in markdown
    assert "4 immutable snapshot(s), 1234 bytes" in markdown
    assert "2 unsupported" in markdown


def test_render_report_includes_body_references_and_audit() -> None:
    report, evidence = _report()
    markdown = render_report(report, evidence)

    assert "## References" in markdown
    assert "## Methodology & Provenance" in markdown
    assert "## Grounding Audit (Appendix)" in markdown
    assert "First source" in markdown
    # Only cited sources appear in the reference list.
    assert "Second source" not in markdown


def test_render_report_falls_back_to_sections_without_a_draft() -> None:
    report, evidence = _report()
    report.draft_markdown = ""
    markdown = render_report(report, evidence)

    assert "The field began with expert systems. [S1]" in markdown


def test_render_report_lists_dangling_citations() -> None:
    report, evidence = _report()
    report.audit.dangling_citations = ["S9"]
    report.audit.gaps = ["How reliable are agents in production?"]
    markdown = render_report(report, evidence)

    assert "`S9`" in markdown
    assert "How reliable are agents in production?" in markdown


def test_reference_list_renders_every_source_when_nothing_is_cited() -> None:
    report, evidence = _report()
    report.draft_markdown = "# Report\n\nNo citations here at all.\n"
    markdown = render_report(report, evidence)

    assert "First source" in markdown
    assert "Second source" in markdown


def test_sources_are_serialisable() -> None:
    source = Source(id="S1", url="https://example.com", title="Title")
    assert source.short() == "[S1] Title"
    assert source.model_dump(mode="json")["kind"] == "unknown"


def test_degraded_report_is_labelled_and_drops_the_executive_summary() -> None:
    report, evidence = _report()
    report.audit.degraded = True
    report.audit.degraded_reasons = ["entailment 0.10 below threshold 0.55"]

    markdown = render_report(report, evidence, degraded=True)

    assert "DEGRADED - not a verified research report" in markdown
    assert "entailment 0.10 below threshold 0.55" in markdown
    # A degraded run must not open with a confident executive summary.
    assert "## Executive Summary" not in markdown
    assert "## Origins" in markdown


def test_methodology_names_a_track_without_evidence_as_not_covered() -> None:
    report, evidence = _report()
    report.sections.append(
        TrackResult(
            track="safety",
            title="Safety, Security and Governance",
            section_markdown="",
            warnings=["Track failed: no evidence-backed finding was produced"],
        )
    )

    markdown = render_report(report, evidence, degraded=True)

    assert "**Safety** - not covered (no evidence-backed finding)" in markdown
