"""Rendering of the final research report."""

from __future__ import annotations

from datetime import UTC, datetime

from .evidence import EvidenceStore
from .schemas import ResearchReport

__all__ = ["render_report", "render_front_matter"]


def render_front_matter(report: ResearchReport) -> str:
    """YAML front matter describing the run."""
    lines = [
        "---",
        f"title: {_yaml(report.title)}",
        f"domain: {_yaml(report.domain)}",
        f"topic: {_yaml(report.topic)}",
        f"generated: {report.generated_at}",
        f"rounds: {report.rounds}",
        f"grounding_score: {report.audit.grounded_ratio:.3f}",
        f"entailment_score: {report.audit.entailment_ratio:.3f}",
        f"audit_status: {report.audit.status}",
        f"audit_verdict: {report.audit.verdict}",
        f"sources: {len(report.sources)}",
        f"findings_recorded: {report.metrics.get('findings_recorded', 0)}",
        f"claims_in_report: {report.metrics.get('claims_in_report', report.audit.total_claims)}",
        f"claims_cited: {report.metrics.get('claims_cited', report.audit.cited_claims)}",
        f"coverage: {_yaml(_coverage(report))}",
        "---",
        "",
    ]
    return "\n".join(lines)


def _coverage(report: ResearchReport) -> str:
    value = report.metrics.get("coverage")
    if value:
        return str(value)
    covered = report.metrics.get("tracks_covered")
    planned = report.metrics.get("tracks_planned")
    if covered is not None and planned is not None:
        return f"{covered}/{planned}"
    return "unknown"


def _yaml(value: str) -> str:
    return f'"{str(value).replace(chr(34), chr(39))}"'


def render_report(
    report: ResearchReport,
    evidence: EvidenceStore,
    *,
    degraded: bool | None = None,
) -> str:
    """Render the complete Markdown report.

    The body comes from the editor agent; the reference list, methodology and
    audit appendix are generated from the run's own records so they can never
    drift away from what actually happened.
    """
    if degraded is None:
        degraded = report.audit.status != "verified"
    parts: list[str] = [render_front_matter(report)]
    if degraded:
        reasons = "; ".join(report.audit.degraded_reasons) or (
            "the audit did not verify this report"
        )
        parts.append(
            "> **DEGRADED - not a verified research report.** "
            f"{reasons}. Treat every claim below as unverified.\n"
        )

    body = _body_text(report)
    if degraded:
        # A degraded run must not open with a confident executive summary.
        body = _strip_executive_summary(body)

    parts.append(body.strip())
    parts.append("")

    used = evidence.extract_citations(body)
    reference_ids = used or [source.id for source in report.sources]
    parts.append("## References")
    parts.append("")
    parts.append(evidence.render_reference_list(reference_ids))
    parts.append("")

    parts.append(_render_methodology(report, evidence))
    parts.append("")
    parts.append(_render_audit(report))

    return "\n".join(parts).strip() + "\n"


def _strip_executive_summary(body: str) -> str:
    """Remove the executive summary from a body that is not verified."""
    marker = "## Executive Summary"
    if marker not in body:
        return body
    head, tail = body.split(marker, 1)
    for separator in ("\n## ", "\n# "):
        if separator in tail:
            rest = tail.split(separator, 1)[1]
            return head.rstrip() + "\n\n" + separator.strip() + " " + rest
    return head.rstrip()


def _body_text(report: ResearchReport) -> str:
    """The report body is the editor's draft; fall back to the sections."""
    draft = report.draft_markdown
    if draft and draft.strip():
        return draft
    return "\n\n".join(
        section.section_markdown for section in report.sections if section.section_markdown
    )


def _render_methodology(report: ResearchReport, evidence: EvidenceStore) -> str:
    metrics = report.metrics or {}
    tools = metrics.get("tool_calls_by_tool", {}) or {}
    tool_lines = "\n".join(f"- `{name}`: {count} calls" for name, count in sorted(tools.items()))
    sections = "\n".join(
        (
            f"- **{section.track.title()}** - not covered (no evidence-backed finding)"
            if section.warnings
            else f"- **{section.track.title()}** - {len(section.findings)} findings, "
            f"{len(section.section_markdown.split())} words"
        )
        for section in report.sections
    )
    return f"""## Methodology & Provenance

This report was produced by **Mars**, an Agentic AI-only multi-agent research
system. The run
took {metrics.get('wall_clock_seconds', 'n/a')} seconds across {report.rounds} round(s)
and used {metrics.get('llm_calls', 0)} model calls
({metrics.get('total_tokens', 0)} tokens, ~${metrics.get('estimated_cost_usd', 0):.4f}).

### Pipeline

1. A **research director** decomposed the topic into independent tracks.
2. Each track was researched in parallel by a **researcher** (live web, papers,
   encyclopaedia and practitioner sources), then converted into structured
   findings by an **evidence analyst**, then written up by a **section writer**.
3. An **editor-in-chief** assembled the sections into the report above.
4. An independent **grounding auditor** checked every citation against the
   source registry and named the remaining gaps.

### Sections

{sections}

### Tools used

{tool_lines or "- (no external tool calls recorded)"}

### Sources

{metrics.get('sources', 0)} sources registered
({metrics.get('findings_recorded', 0)} findings recorded in the claim ledger;
{metrics.get('claims_cited', 0)} of {metrics.get('claims_in_report', 0)} report
claims carry a verifiable citation).
"""


def _render_audit(report: ResearchReport) -> str:
    audit = report.audit
    lines = [
        "## Grounding Audit (Appendix)",
        "",
        f"- Status: **{audit.status}**",
        f"- Coverage: **{_coverage(report)}** planned tracks",
        f"- Grounding score: **{audit.grounded_ratio:.0%}** "
        f"(threshold {report.metrics.get('grounding_threshold', 0.85):.0%})",
        f"- Entailment score: **{audit.entailment_ratio:.0%}** "
        f"(threshold {report.metrics.get('entailment_threshold', 0.55):.0%})",
        f"- Entailment checks: {audit.entailment_supported} supported, "
        f"{audit.entailment_partial} partial, {audit.entailment_unsupported} unsupported, "
        f"{audit.entailment_unverified} unverified (of {audit.entailment_total})",
        f"- Audit method: {audit.audit_method}",
        f"- Verdict: **{audit.verdict}**",
        f"- Cited claims: {audit.cited_claims} of {audit.total_claims}",
    ]
    if audit.degraded_reasons:
        lines.append("- Why this report is not marked verified:")
        lines.extend(f"  - {reason}" for reason in audit.degraded_reasons)
    archived = report.metrics.get("evidence_archive", {}) or {}
    if archived:
        lines.append(
            f"- Evidence archive: {archived.get('entries', 0)} immutable snapshot(s), "
            f"{archived.get('bytes', 0)} bytes"
        )
    if audit.dangling_citations:
        lines.append(
            "- Dangling citations: " + ", ".join(f"`{c}`" for c in audit.dangling_citations)
        )
    if audit.unsupported_claims:
        lines.append("- Claims flagged as insufficiently supported:")
        lines.extend(f"  - {claim}" for claim in audit.unsupported_claims)
    if audit.gaps:
        lines.append("- Open gaps carried forward:")
        lines.extend(f"  - {gap}" for gap in audit.gaps)
    if audit.notes:
        lines.append(f"- Auditor's note: {audit.notes}")
    lines.append("")
    lines.append(
        f"_Audit generated {datetime.now(UTC).replace(microsecond=0).isoformat()}._"
    )
    return "\n".join(lines)
