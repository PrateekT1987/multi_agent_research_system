"""Typed data structures shared by every agent in the system.

The agents communicate through these models rather than free text whenever a
structured payload is required.  That gives us three things for free:

* validation (a malformed ``Finding`` fails loudly instead of silently
  poisoning the report),
* a serialisable record of the run (every intermediate artefact is dumped to
  JSON next to the final report), and
* the raw material for the grounding audit, which cross-checks claims against
  the source registry.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .domain import DOMAIN

__all__ = [
    "Source",
    "Claim",
    "TrackPlan",
    "ResearchPlan",
    "TrackResult",
    "FindingBundle",
    "AuditBundle",
    "AuditResult",
    "EvaluationResult",
    "TimelineEntry",
    "ResearchReport",
    "utcnow_iso",
    "parse_json_block",
]

CitationId = str


def utcnow_iso() -> str:
    """Timestamp helper used across artefacts."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class Source(BaseModel):
    """A single retrievable piece of evidence."""

    id: CitationId
    url: str
    title: str
    snippet: str = ""
    kind: Literal["web", "paper", "wiki", "community", "unknown"] = "unknown"
    provider: str = "unknown"
    published: str | None = None
    retrieved_at: str = Field(default_factory=utcnow_iso)
    # Immutable provenance: the archived snapshot plus the response metadata.
    content_hash: str = ""
    archive_path: str = ""
    final_url: str = ""
    content_type: str = ""
    http_status: int = 0
    quality: float = 0.0
    license: str = "unknown"
    paywalled: bool = False
    injection_flags: list[str] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def _default_title(cls, value: str) -> str:
        value = (value or "").strip()
        return value or "Untitled source"

    @field_validator("quality")
    @classmethod
    def _quality_range(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def short(self) -> str:
        return f"[{self.id}] {self.title}"


class Claim(BaseModel):
    """A factual assertion recorded during research, with its evidence."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    text: str
    evidence: list[CitationId] = Field(default_factory=list)
    confidence: float = 0.5
    track: str = "general"
    kind: Literal["fact", "trend", "forecast", "controversy"] = "fact"
    # Result of the deterministic entailment check against the cited sources.
    verification: Literal["supported", "partial", "unsupported", "unverified"] = "unverified"
    verification_score: float = 0.0
    verification_notes: str = ""
    verified_at: str | None = None

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class TrackPlan(BaseModel):
    """The research brief for a single track."""

    track: str
    title: str = ""  # Human-readable title for the report section
    objective: str
    key_questions: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """The director's decomposition of the topic into track briefs."""

    topic: str
    domain: str = DOMAIN
    framing: str = ""
    tracks: list[TrackPlan] = Field(default_factory=list)
    generated_at: str = Field(default_factory=utcnow_iso)


class TrackResult(BaseModel):
    """Everything one research track produced."""

    track: str
    title: str = ""
    dossier: str = ""
    section_markdown: str = ""
    findings: list[Claim] = Field(default_factory=list)
    source_ids: list[CitationId] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FindingBundle(BaseModel):
    """Structured output of the analyst agent."""

    track: str
    summary: str = ""
    findings: list[Claim] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class AuditBundle(BaseModel):
    """Structured output of the grounding auditor."""

    grounded_ratio: float = 0.0
    cited_claims: int = 0
    total_claims: int = 0
    unsupported_claims: list[str] = Field(default_factory=list)
    dangling_citations: list[str] = Field(default_factory=list)
    uncited_paragraphs: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    verdict: str = "unreviewed"
    notes: str = ""
    entailment_ratio: float = 0.0
    entailment_supported: int = 0
    entailment_total: int = 0
    degraded: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)



class TimelineEntry(BaseModel):
    """A dated event used to build the report's chronology."""

    date: str
    event: str
    significance: str = ""
    source_ids: list[CitationId] = Field(default_factory=list)


class AuditResult(BaseModel):
    """Output of the grounding / coverage audit."""

    grounded_ratio: float = 0.0
    cited_claims: int = 0
    total_claims: int = 0
    unsupported_claims: list[str] = Field(default_factory=list)
    dangling_citations: list[str] = Field(default_factory=list)
    uncited_paragraphs: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    verdict: str = "unreviewed"
    notes: str = ""
    # Deterministic entailment audit against archived source text.
    entailment_ratio: float = 0.0
    entailment_total: int = 0
    entailment_supported: int = 0
    entailment_partial: int = 0
    entailment_unsupported: int = 0
    entailment_unverified: int = 0
    claim_verification: dict[str, int] = Field(default_factory=dict)
    audit_method: str = "mechanical"
    # Explicitly distinguish a verified report from a degraded one.
    verified: bool = False
    degraded: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.grounded_ratio >= 0.85 and not self.dangling_citations

    @property
    def status(self) -> str:
        if self.verified:
            return "verified"
        return "degraded" if self.degraded else "unreviewed"


class EvaluationResult(BaseModel):
    """Independent, bounded quality evaluation of a completed report."""

    score: float = 0.0
    verdict: Literal["pass", "revise", "fail", "unreviewed"] = "unreviewed"
    rationale: str = ""
    issues: list[str] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def _score_range(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class ResearchReport(BaseModel):
    """The final, rendered research artefact."""

    topic: str
    domain: str = DOMAIN
    title: str
    generated_at: str = Field(default_factory=utcnow_iso)
    executive_summary: str = ""
    draft_markdown: str = ""
    sections: list[TrackResult] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    audit: AuditResult = Field(default_factory=AuditResult)
    metrics: dict[str, Any] = Field(default_factory=dict)
    rounds: int = 1
    methodology: str = ""


def parse_json_block(text: str) -> Any:
    """Extract JSON from an LLM response.

    Models like to wrap JSON in prose or triple backticks, so we try the
    strict parse first and then fall back to the outermost ``{...}`` / ``[...]``
    span in the text.
    """
    import json

    candidate = (text or "").strip()
    if not candidate:
        raise ValueError("empty response")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError("could not parse JSON from response")


TrackResult.model_rebuild()
