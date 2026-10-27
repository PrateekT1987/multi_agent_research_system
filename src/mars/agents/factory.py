"""Factory that turns profiles into CrewAI agents and crews.

Each stage of the pipeline is a single-agent crew whose prompt is fully
materialised before the kickoff.  The pipeline - not CrewAI - wires the stages
together, which keeps the data flow between agents explicit and inspectable
(you can always ask "what exactly did the writer see?").

The agents themselves are ordinary CrewAI agents: they use CrewAI's tool
execution loop, its native function calling, and its structured-output
plumbing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llm import BaseLLM
from crewai.tools import BaseTool

from ..config import Settings
from ..context import ResearchContext
from ..domain import (
    DEFAULT_DOMAIN_TRACKS,
    DOMAIN,
    DOMAIN_TRACK_KEYS,
    canonical_track,
    ensure_domain_topic,
    scope_query,
)
from ..schemas import AuditBundle, FindingBundle, ResearchPlan
from .profiles import (
    ANALYST_PROFILE,
    AUDITOR_PROFILE,
    DIRECTOR_PROFILE,
    EDITOR_PROFILE,
    GAP_RESEARCHER_PROFILE,
    RESEARCHER_PROFILE,
    WRITER_PROFILE,
    AgentProfile,
)
from .tasks import (
    analyst_task,
    auditor_task,
    director_task,
    editor_task,
    gap_researcher_task,
    researcher_task,
    writer_task,
)

__all__ = [
    "build_agent",
    "run_director",
    "run_research",
    "run_analysis",
    "run_writing",
    "run_editing",
    "run_audit",
    "run_gap_research",
    "crew_to_markdown",
    "plan_to_dict",
    "dossier_problem",
    "section_citation_problem",
]

#: Markers a model emits when it writes a tool call as prose instead of
#: returning a native tool call.  CrewAI treats the message as the final
#: answer, so the "dossier" is a request to search that was never executed.
_TOOL_CALL_MARKERS = ("<tool_call>", "<function=", "<parameter=", '"tool_calls"')
_CITATION_RE = re.compile(r"\[S\d+\]")
#: A dossier shorter than this cannot carry a track's evidence.
_MIN_DOSSIER_WORDS = 60


def dossier_problem(dossier: str) -> str | None:
    """Explain why a research dossier is unusable, or return ``None``.

    A model that is weak at native tool calling sometimes answers with a tool
    call in plain text.  CrewAI accepts that as the final answer, the analyst
    finds nothing to structure, and the writer fills the vacuum with prose from
    its own priors.  Rejecting the dossier here is what stops an evidence-free
    section from ever reaching the report.
    """
    text = (dossier or "").strip()
    if not text:
        return "the dossier was empty"
    lowered = text.lower()
    if any(marker in lowered for marker in _TOOL_CALL_MARKERS):
        return "the researcher returned an unexecuted tool call instead of findings"
    if not _CITATION_RE.search(text):
        return "the dossier cited no sources"
    if len(text.split()) < _MIN_DOSSIER_WORDS:
        return f"the dossier was too short ({len(text.split())} words)"
    return None


def section_citation_problem(
    section: str, evidence: Any, allowed_ids: Iterable[str] | None = None
) -> str | None:
    """Explain why a written section cannot enter the report, or ``None``."""
    text = (section or "").strip()
    if not text:
        return "the writer returned an empty section"
    cited = evidence.extract_citations(text)
    if not cited:
        return "the section contained no [S#] citations"
    dangling = evidence.dangling_citations(text)
    if dangling:
        return "the section cited unregistered sources: " + ", ".join(dangling)
    flagged = [
        cid
        for cid in cited
        if (source := evidence.get(cid)) is not None and source.injection_flags
    ]
    if flagged:
        return (
            "the section cited sources carrying prompt-injection flags: "
            + ", ".join(flagged)
        )
    if allowed_ids is not None:
        allowed = {str(item).upper() for item in allowed_ids}
        outside = [cid for cid in cited if cid not in allowed]
        if outside:
            return (
                "the section cited sources whose snippets it was not given: "
                + ", ".join(outside)
            )
    return None


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------
def build_agent(
    profile: AgentProfile,
    llm: BaseLLM,
    *,
    tools: list[BaseTool] | None = None,
    max_iter: int = 12,
    verbose: bool = False,
) -> Agent:
    """Create a CrewAI agent from a profile."""
    return Agent(
        role=profile.role,
        goal=profile.goal,
        backstory=profile.backstory,
        llm=llm,
        tools=tools or [],
        max_iter=max_iter,
        allow_delegation=False,
        verbose=verbose,
        inject_date=True,
    )


def _single_agent_crew(
    agent: Agent,
    description: str,
    expected_output: str,
    *,
    name: str,
    output_pydantic: type[Any] | None = None,
    verbose: bool = False,
) -> Crew:
    task = Task(
        name=name,
        description=description,
        expected_output=expected_output,
        agent=agent,
        output_pydantic=output_pydantic,
    )
    return Crew(
        name=name,
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=verbose,
        memory=False,
        planning=False,
    )


def crew_to_markdown(output: Any) -> str:
    """Best-effort extraction of markdown text from a CrewAI output."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    raw = getattr(output, "raw", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    pydantic_output = getattr(output, "pydantic", None)
    if pydantic_output is not None:
        if hasattr(pydantic_output, "model_dump_json"):
            return pydantic_output.model_dump_json()
        if isinstance(pydantic_output, str):
            return pydantic_output
    if isinstance(output, dict):
        for key in ("raw", "markdown", "text", "output"):
            if isinstance(output.get(key), str):
                return output[key].strip()
    return json.dumps(output, default=str)


def plan_to_dict(plan: ResearchPlan | None) -> dict[str, Any] | None:
    """Convert a research plan into a plain dict for prompt interpolation."""
    if plan is None:
        return None
    data = plan.model_dump()
    tracks = data.get("tracks") or []
    return {"track": tracks[0], "tracks": tracks} if tracks else None


# ---------------------------------------------------------------------------
# Stage 0 - director
# ---------------------------------------------------------------------------
def run_director(
    llm: BaseLLM,
    topic: str,
    tracks: list[str] | None = None,
    *,
    verbose: bool = False,
) -> ResearchPlan:
    """Produce the Agentic AI research plan."""
    topic = ensure_domain_topic(topic)
    # The Director may only choose from the fixed domain taxonomy.
    allowed = list(tracks or DOMAIN_TRACK_KEYS)
    agent = build_agent(DIRECTOR_PROFILE, llm, verbose=verbose)
    description, expected = director_task(topic, allowed)
    crew = _single_agent_crew(
        agent,
        description,
        expected,
        name="director",
        output_pydantic=ResearchPlan,
        verbose=verbose,
    )
    output = crew.kickoff()
    plan = getattr(output, "pydantic", None)
    if isinstance(plan, ResearchPlan):
        return plan
    return _recover_plan(crew_to_markdown(output), topic, allowed)


def _recover_plan(raw: str, topic: str, tracks: list[str]) -> ResearchPlan:
    """Best-effort recovery of a plan when structured output failed."""
    from ..schemas import TrackPlan, parse_json_block

    topic = ensure_domain_topic(topic)
    try:
        payload = parse_json_block(raw)
        if isinstance(payload, dict):
            data = payload.get("tracks") or []
            if data:
                recovered: list[TrackPlan] = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    canonical = canonical_track(str(item.get("track", "")))
                    if canonical is None:
                        continue
                    item = {**item, "track": canonical.key}
                    recovered.append(TrackPlan.model_validate(item))
                if recovered:
                    return ResearchPlan(
                        topic=topic,
                        domain=DOMAIN,
                        framing=payload.get("framing", ""),
                        tracks=recovered,
                    )
    except Exception:
        pass

    fallback_keys = [t for t in tracks if canonical_track(t) is not None]
    if not fallback_keys:
        fallback_keys = list(DEFAULT_DOMAIN_TRACKS)
    fallback = [
        TrackPlan(
            track=track,
            title=canonical_track(track).title if canonical_track(track) else track,
            objective=f"Establish the {track} of {topic} within Agentic AI.",
            key_questions=[],
            search_queries=[scope_query(f"{topic} {track}")],
            keywords=[topic, track, DOMAIN],
        )
        for track in fallback_keys
    ]
    return ResearchPlan(
        topic=topic,
        domain=DOMAIN,
        framing="Fallback plan (structured output unavailable).",
        tracks=fallback,
    )


# ---------------------------------------------------------------------------
# Stage 1 - research
# ---------------------------------------------------------------------------
def run_research(
    context: ResearchContext,
    llm: BaseLLM,
    topic: str,
    track: str,
    plan: dict[str, Any] | None = None,
    *,
    gap_questions: list[str] | None = None,
    repair: bool = False,
    verbose: bool = False,
) -> str:
    """Research one track and return the dossier."""
    settings: Settings = context.settings
    from ..tools import build_toolset

    agent = build_agent(
        RESEARCHER_PROFILE,
        llm,
        tools=build_toolset(context, track=track),
        max_iter=settings.max_tool_iterations,
        verbose=verbose,
    )
    description, expected = researcher_task(
        topic, track, plan, gap_questions, repair=repair
    )
    crew = _single_agent_crew(agent, description, expected, name=f"research_{track}", verbose=verbose)
    return crew_to_markdown(crew.kickoff())


# ---------------------------------------------------------------------------
# Stage 2 - analysis
# ---------------------------------------------------------------------------
def run_analysis(
    llm: BaseLLM,
    track: str,
    dossier: str,
    *,
    verbose: bool = False,
) -> FindingBundle:
    """Turn a dossier into structured findings."""
    agent = build_agent(ANALYST_PROFILE, llm, verbose=verbose)
    description, expected = analyst_task(track, dossier)
    crew = _single_agent_crew(
        agent,
        description,
        expected,
        name=f"analyse_{track}",
        output_pydantic=FindingBundle,
        verbose=verbose,
    )
    output = crew.kickoff()
    bundle = getattr(output, "pydantic", None)
    if isinstance(bundle, FindingBundle):
        return bundle
    return _recover_findings(crew_to_markdown(output), track)


def _recover_findings(raw: str, track: str) -> FindingBundle:
    from ..schemas import parse_json_block

    try:
        payload = parse_json_block(raw)
        if isinstance(payload, dict):
            return FindingBundle.model_validate(
                {
                    "track": payload.get("track", track),
                    "summary": payload.get("summary", ""),
                    "findings": payload.get("findings", []),
                    "timeline": payload.get("timeline", []),
                    "open_questions": payload.get("open_questions", []),
                }
            )
    except Exception:
        pass
    return FindingBundle(
        track=track,
        summary="Structured analysis could not be parsed; the raw dossier is used instead.",
        findings=[],
    )


# ---------------------------------------------------------------------------
# Stage 3 - section writing
# ---------------------------------------------------------------------------
def run_writing(
    llm: BaseLLM,
    topic: str,
    track: str,
    dossier: str,
    findings_json: str,
    *,
    title: str | None = None,
    repair: bool = False,
    sources_block: str = "",
    forbidden_ids: list[str] | None = None,
    verbose: bool = False,
) -> str:
    """Write one section of the report."""
    agent = build_agent(WRITER_PROFILE, llm, verbose=verbose)
    description, expected = writer_task(
        topic,
        track,
        dossier,
        findings_json,
        title,
        repair=repair,
        sources_block=sources_block,
        forbidden_ids=forbidden_ids,
    )
    crew = _single_agent_crew(agent, description, expected, name=f"write_{track}", verbose=verbose)
    return crew_to_markdown(crew.kickoff())


# ---------------------------------------------------------------------------
# Stage 4 - editing
# ---------------------------------------------------------------------------
def run_editing(
    llm: BaseLLM,
    topic: str,
    title: str,
    sections_markdown: str,
    findings_json: str,
    timeline_json: str,
    *,
    track_titles: list[str] | None = None,
    verbose: bool = False,
) -> str:
    """Assemble the final report."""
    agent = build_agent(EDITOR_PROFILE, llm, verbose=verbose)
    description, expected = editor_task(
        topic, title, sections_markdown, findings_json, timeline_json, track_titles
    )
    crew = _single_agent_crew(agent, description, expected, name="editor", verbose=verbose)
    return crew_to_markdown(crew.kickoff())


# ---------------------------------------------------------------------------
# Stage 5 - audit
# ---------------------------------------------------------------------------
def run_audit(
    llm: BaseLLM,
    draft: str,
    sources_block: str,
    *,
    verbose: bool = False,
) -> AuditBundle:
    """Audit the draft against the source registry."""
    agent = build_agent(AUDITOR_PROFILE, llm, verbose=verbose)
    description, expected = auditor_task(draft, sources_block)
    crew = _single_agent_crew(
        agent,
        description,
        expected,
        name="auditor",
        output_pydantic=AuditBundle,
        verbose=verbose,
    )
    output = crew.kickoff()
    bundle = getattr(output, "pydantic", None)
    if isinstance(bundle, AuditBundle):
        return bundle
    return _recover_audit(crew_to_markdown(output))


def _recover_audit(raw: str) -> AuditBundle:
    from ..schemas import parse_json_block

    try:
        payload = parse_json_block(raw)
        if isinstance(payload, dict):
            return AuditBundle.model_validate(payload)
    except Exception:
        pass
    return AuditBundle(
        grounded_ratio=0.0,
        verdict="unreviewed",
        notes="The auditor's structured output could not be parsed.",
    )


# ---------------------------------------------------------------------------
# Stage 6 - gap closing
# ---------------------------------------------------------------------------
def run_gap_research(
    context: ResearchContext,
    llm: BaseLLM,
    topic: str,
    gaps: list[str],
    *,
    verbose: bool = False,
) -> str:
    """Research specific gaps flagged by the auditor."""
    settings: Settings = context.settings
    from ..tools import build_toolset

    agent = build_agent(
        GAP_RESEARCHER_PROFILE,
        llm,
        tools=build_toolset(context, track="gaps"),
        max_iter=settings.max_tool_iterations,
        verbose=verbose,
    )
    description, expected = gap_researcher_task(topic, gaps)
    crew = _single_agent_crew(agent, description, expected, name="gap_researcher", verbose=verbose)
    return crew_to_markdown(crew.kickoff())
