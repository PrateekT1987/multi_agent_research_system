"""Formal behaviour contracts for every agent role.

A profile describes an agent's persona.  A contract describes what the agent is
allowed to receive, what it must produce, and how it fails.  The pipeline
validates every profile against a contract before a run starts, and the contract
manifest is recorded in every report and run manifest so a behaviour change is
auditable.

Contracts are deliberately written in code next to the profiles: they are part
of the release, reviewed in the same pull request, and exercised by
``tests/test_contracts.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..schemas import AuditBundle, EvaluationResult, FindingBundle, ResearchPlan, TrackResult
from .profiles import PROFILES

__all__ = [
    "CONTRACTS",
    "CONTRACT_VERSION",
    "AgentContract",
    "ContractError",
    "contract_for",
    "contract_manifest",
    "schema_model",
    "validate_contracts",
]

CONTRACT_VERSION = "1.0.0"


class ContractError(ValueError):
    """An agent has no contract, or its contract is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class AgentContract:
    """The interface an agent implementation must honour."""

    key: str
    version: str
    responsibilities: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    output_schema: str
    failure_modes: tuple[str, ...]
    invariants: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Pydantic models that back a contract's structured output.  Markdown outputs
# are named but intentionally have no model: they are prose with citation
# invariants enforced by the auditor.
SCHEMA_MODELS: dict[str, type[Any]] = {
    "ResearchPlan": ResearchPlan,
    "FindingBundle": FindingBundle,
    "AuditBundle": AuditBundle,
    "EvaluationResult": EvaluationResult,
    "TrackResult": TrackResult,
}

_MARKDOWN_SCHEMAS = {"Dossier", "SectionMarkdown", "ReportMarkdown", "GapNotes"}

CONTRACTS: dict[str, AgentContract] = {
    "director": AgentContract(
        key="director",
        version=CONTRACT_VERSION,
        responsibilities=(
            "Turn one Agentic AI subtopic into a bounded, non-overlapping research plan.",
            "Choose 4-6 tracks from the fixed taxonomy and write answerable questions.",
        ),
        inputs=("topic (domain-locked)", "canonical track menu", "maximum track count"),
        outputs=("ResearchPlan", "per-track objectives, questions, queries, keywords, pitfalls"),
        output_schema="ResearchPlan",
        failure_modes=(
            "invalid JSON or a plan that fails schema validation",
            "invented or off-taxonomy track keys",
            "questions or queries that leave the Agentic AI domain",
            "more tracks than the configured maximum",
        ),
        invariants=(
            "every track is canonical",
            "the topic is inside the Agentic AI domain",
            "the plan is normalised again by the pipeline before research",
        ),
    ),
    "researcher": AgentContract(
        key="researcher",
        version=CONTRACT_VERSION,
        responsibilities=(
            "Gather primary evidence for one assigned track using the retrieval tools.",
            "Record every load-bearing claim with the source ids that support it.",
        ),
        inputs=("topic", "track brief", "planned queries", "optional gap questions"),
        outputs=("Markdown dossier", "evidence-store records", "source ids"),
        output_schema="Dossier",
        failure_modes=(
            "tool or network failure",
            "off-domain or paywalled sources",
            "claims recorded without a source id",
            "invented [S#] ids",
        ),
        invariants=(
            "only Agentic AI evidence is recorded",
            "every recorded finding cites an existing source id",
            "missing evidence is reported as a gap, never fabricated",
        ),
    ),
    "analyst": AgentContract(
        key="analyst",
        version=CONTRACT_VERSION,
        responsibilities=(
            "Convert a dossier into deduplicated findings with calibrated confidence.",
            "Separate facts, trends, forecasts and controversies.",
        ),
        inputs=("track", "dossier markdown"),
        outputs=("FindingBundle", "timeline", "open questions", "track summary"),
        output_schema="FindingBundle",
        failure_modes=(
            "invalid JSON or schema failure",
            "invented evidence ids",
            "duplicate or off-domain findings",
            "forecasts presented as facts",
        ),
        invariants=(
            "every finding cites ids present in the dossier",
            "confidence is within 0.0-1.0",
            "no new claims are introduced",
        ),
    ),
    "writer": AgentContract(
        key="writer",
        version=CONTRACT_VERSION,
        responsibilities=(
            "Write one self-contained report section from the findings and dossier.",
            "Cite every substantive claim with a [S#] id.",
        ),
        inputs=("topic", "track", "findings JSON", "dossier", "section title"),
        outputs=("Markdown section",),
        output_schema="SectionMarkdown",
        failure_modes=(
            "uncited or weakly cited prose",
            "raw URLs or non-[S#] citation formats",
            "domain drift into general AI or unrelated fields",
            "missing or duplicated heading",
        ),
        invariants=(
            "the section starts with the assigned heading",
            "all citations use [S#] and exist in the registry",
            "evidence gaps are stated rather than filled",
        ),
    ),
    "editor": AgentContract(
        key="editor",
        version=CONTRACT_VERSION,
        responsibilities=(
            "Assemble section drafts into the final report with the mandated structure.",
            "Preserve every citation and remove duplication across sections.",
        ),
        inputs=("topic", "title", "section drafts", "findings JSON", "timeline JSON", "track titles"),
        outputs=("complete Markdown report",),
        output_schema="ReportMarkdown",
        failure_modes=(
            "dropped or rewritten citations",
            "missing mandated sections",
            "duplicated passages across sections",
            "length or domain drift",
        ),
        invariants=(
            "the structure and section order are fixed",
            "no new citation ids are introduced",
            "the report stays inside Agentic AI",
        ),
    ),
    "auditor": AgentContract(
        key="auditor",
        version=CONTRACT_VERSION,
        responsibilities=(
            "Check citations against the source registry and test claim support.",
            "Name concrete coverage gaps and overclaims.",
        ),
        inputs=("draft markdown", "rendered source registry"),
        outputs=("AuditBundle", "grounded ratio", "dangling citations", "gaps", "verdict"),
        output_schema="AuditBundle",
        failure_modes=(
            "invalid JSON or schema failure",
            "missed dangling citations",
            "accepting an overclaim",
            "vague, unactionable gaps",
        ),
        invariants=(
            "every [S#] in the draft is checked",
            "gaps are specific and answerable",
            "the verdict is conservative when evidence is thin",
        ),
    ),
    "gap_researcher": AgentContract(
        key="gap_researcher",
        version=CONTRACT_VERSION,
        responsibilities=(
            "Close named audit gaps with targeted searches and evidence recording.",
            "Report a confirmed absence of evidence instead of speculating.",
        ),
        inputs=("topic", "ordered gap questions"),
        outputs=("gap findings with citations", "remaining gaps"),
        output_schema="GapNotes",
        failure_modes=(
            "off-domain gap filling",
            "speculation presented as evidence",
            "ignoring a gap",
        ),
        invariants=(
            "every gap gets an answer or an explicit no-evidence statement",
            "only [S#] citations are used",
            "domain boundaries are respected",
        ),
    ),
}


def contract_for(key: str) -> AgentContract:
    """Return the contract for an agent key, or raise."""
    try:
        return CONTRACTS[key]
    except KeyError as exc:
        raise ContractError(f"no behaviour contract is defined for agent {key!r}") from exc


def schema_model(name: str) -> type[Any] | None:
    """Resolve a structured-output schema name to its Pydantic model."""
    return SCHEMA_MODELS.get(name)


def validate_contracts() -> list[str]:
    """Return every inconsistency between profiles and contracts."""
    errors: list[str] = []
    missing = sorted(set(PROFILES) - set(CONTRACTS))
    orphaned = sorted(set(CONTRACTS) - set(PROFILES))
    if missing:
        errors.append(f"agents without a contract: {', '.join(missing)}")
    if orphaned:
        errors.append(f"contracts without an agent profile: {', '.join(orphaned)}")

    for key, contract in CONTRACTS.items():
        if contract.key != key:
            errors.append(f"contract key {contract.key!r} does not match registry key {key!r}")
        if not contract.version:
            errors.append(f"contract {key!r} has no version")
        for field_name in (
            "responsibilities",
            "inputs",
            "outputs",
            "failure_modes",
            "invariants",
        ):
            if not getattr(contract, field_name):
                errors.append(f"contract {key!r} has no {field_name}")
        if contract.output_schema not in SCHEMA_MODELS and contract.output_schema not in _MARKDOWN_SCHEMAS:
            errors.append(
                f"contract {key!r} names unknown output schema {contract.output_schema!r}"
            )
    return errors


def contract_manifest() -> dict[str, Any]:
    """Serialise all contracts for a report or run manifest."""
    return {
        "version": CONTRACT_VERSION,
        "agents": {key: contract.as_dict() for key, contract in sorted(CONTRACTS.items())},
    }
