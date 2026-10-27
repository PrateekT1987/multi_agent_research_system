"""Mars - an Agentic AI-only multi-agent research system.

Public API::

    from mars import ResearchPipeline, build_settings

    settings = build_settings(topic="agentic AI")
    result = ResearchPipeline(settings).run()
    print(result.report_markdown)
"""

from __future__ import annotations

import os

# Keep CrewAI quiet and telemetry-free by default.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_STORAGE_DIR", os.path.expanduser("~/.cache/mars/crewai"))

from .config import Settings, build_settings
from .context import ResearchContext
from .domain import DOMAIN, DOMAIN_TRACKS, DomainError, ensure_domain_topic
from .evidence import EvidenceStore
from .llm import OpenRouterLLM, build_llm
from .pipeline import PipelineResult, ResearchPipeline
from .reporting import render_report
from .schemas import (
    AuditResult,
    Claim,
    ResearchPlan,
    ResearchReport,
    Source,
    TimelineEntry,
    TrackResult,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "Settings",
    "build_settings",
    "DOMAIN",
    "DOMAIN_TRACKS",
    "DomainError",
    "ensure_domain_topic",
    "ResearchContext",
    "EvidenceStore",
    "OpenRouterLLM",
    "build_llm",
    "ResearchPipeline",
    "PipelineResult",
    "render_report",
    "AuditResult",
    "Claim",
    "ResearchPlan",
    "ResearchReport",
    "Source",
    "TimelineEntry",
    "TrackResult",
]
