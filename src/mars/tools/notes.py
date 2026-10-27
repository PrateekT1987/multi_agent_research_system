"""Tools for recording findings and reading the evidence registry.

These are what make the multi-agent system auditable: the researcher states
what it believes and which sources back it, and the auditor later checks the
finished report against that ledger.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base import MarsTool

__all__ = [
    "RecordFindingTool",
    "ListSourcesTool",
    "ReadFindingsTool",
    "RecordFindingArgs",
]


class RecordFindingArgs(BaseModel):
    claim: str = Field(description="The factual statement, written as a complete sentence.")
    source_ids: str = Field(
        default="",
        description="Comma-separated citation ids supporting the claim, e.g. 'S1,S4'.",
    )
    confidence: float = Field(default=0.6, description="Confidence between 0 and 1.")
    kind: str = Field(
        default="fact",
        description="One of: fact, trend, forecast, controversy.",
    )


class RecordFindingTool(MarsTool):
    """Record a finding together with the sources that support it.

    Call this for every significant claim you intend to use in your write-up.
    The claim ledger is later used to audit whether the final report is
    properly grounded in evidence.
    """

    name: str = "record_finding"
    description: str = (
        "Record an important finding with the citation ids that support it. "
        "Use this for each claim you plan to include in the final section."
    )
    args_schema: type[BaseModel] = RecordFindingArgs
    track: str = "general"

    def _run(
        self,
        claim: str = "",
        source_ids: str = "",
        confidence: float = 0.6,
        kind: str = "fact",
        **_: Any,
    ) -> str:
        if not claim.strip():
            return "Error: claim must not be empty."
        ids = [s.strip().upper() for s in source_ids.replace(";", ",").split(",") if s.strip()]
        known = set(self.evidence.source_ids())
        missing = [cid for cid in ids if cid not in known]
        record = self.evidence.add_claim(
            claim,
            ids,
            confidence=confidence,
            track=self.track,
            kind=kind if kind in {"fact", "trend", "forecast", "controversy"} else "fact",
        )
        warning = ""
        if missing:
            warning = (
                f" Warning: {', '.join(missing)} are not registered sources; "
                "search or fetch them first so the citation resolves."
            )
        if not ids:
            warning += " Warning: no source ids supplied - this finding will be uncited."
        return (
            f"Recorded {record.id} (confidence {record.confidence:.2f}, "
            f"evidence: {', '.join(record.evidence) or 'none'}).{warning}"
        )


class ListSourcesTool(MarsTool):
    """List the sources collected so far."""

    name: str = "list_sources"
    description: str = (
        "List every source registered so far with its citation id, title and URL. "
        "Use it to check which ids are available before citing them."
    )
    args_schema: type[BaseModel] = BaseModel

    def _run(self, **_: Any) -> str:
        sources = self.evidence.sources()
        if not sources:
            return "No sources have been registered yet. Use web_search or fetch_page first."
        lines = [f"{len(sources)} source(s) registered:"]
        for source in sources:
            lines.append(f"[{source.id}] {source.title} - {source.url}")
        return "\n".join(lines)


class ReadFindingsTool(MarsTool):
    """Read the claim ledger."""

    name: str = "read_findings"
    description: str = (
        "Read the findings recorded so far, including the source ids backing each one. "
        "Use it to check your coverage before writing the section."
    )
    args_schema: type[BaseModel] = BaseModel

    def _run(self, **_: Any) -> str:
        claims = self.evidence.claims()
        if not claims:
            return "No findings recorded yet."
        lines = [f"{len(claims)} finding(s) recorded:"]
        for claim in claims:
            evidence = ", ".join(claim.evidence) or "NO EVIDENCE"
            lines.append(
                f"- {claim.text} [{evidence}] (confidence {claim.confidence:.2f}, {claim.kind})"
            )
        return "\n".join(lines)
