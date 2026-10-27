"""Agent definitions and orchestration helpers for Mars."""

from .factory import (
    build_agent,
    dossier_problem,
    plan_to_dict,
    run_analysis,
    run_audit,
    run_director,
    run_editing,
    run_gap_research,
    run_research,
    run_writing,
    section_citation_problem,
)
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

__all__ = [
    "AgentProfile",
    "DIRECTOR_PROFILE",
    "RESEARCHER_PROFILE",
    "ANALYST_PROFILE",
    "WRITER_PROFILE",
    "EDITOR_PROFILE",
    "AUDITOR_PROFILE",
    "GAP_RESEARCHER_PROFILE",
    "build_agent",
    "plan_to_dict",
    "dossier_problem",
    "section_citation_problem",
    "run_director",
    "run_research",
    "run_analysis",
    "run_writing",
    "run_editing",
    "run_audit",
    "run_gap_research",
]
