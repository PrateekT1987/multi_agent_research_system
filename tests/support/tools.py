"""Test-only retrieval tools.

Production no longer ships a curated corpus or any offline/mock provider.  The
hermetic test-suite still needs a deterministic, network-free source of
citable material, so this module provides one.  It is imported only by
``tests/conftest.py``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

from mars.sanitize import untrusted_header
from mars.tools.base import MarsTool, format_sources

__all__ = ["TestCorpusTool", "TestCorpusArgs"]

_TOPICS = (
    ("History of autonomous agents", "Early rule-based agents and expert systems established the idea of goal-directed software."),
    ("Tool-using language models", "Modern agent frameworks let a language model call tools, observe results and iterate."),
    ("Reliability of agentic systems", "Production deployments emphasise evaluation, guardrails and human oversight."),
    ("Economics of agent deployment", "Adoption is concentrated in software engineering and customer support workflows."),
    ("Governance and safety", "Regulators and standards bodies are drafting guidance for autonomous systems."),
    ("Future trajectories", "Forecasts disagree on the pace of autonomy but agree that tool use is central."),
)


class TestCorpusArgs(BaseModel):
    query: str = Field(description="Free-text query.")
    max_results: int = Field(default=10, description="How many entries to return (1-20).")


class TestCorpusTool(MarsTool):
    """A deterministic stand-in for a retrieval backend (tests only)."""

    name: str = "query_corpus"
    description: str = (
        "Search a deterministic test corpus. Returns dated, citable entries. "
        "This tool exists only in the hermetic test harness."
    )
    args_schema: type[BaseModel] = TestCorpusArgs

    def _run(self, query: str = "", max_results: int = 10, **_: Any) -> str:
        limit = max(1, min(int(max_results or 10), 20))
        digest = hashlib.sha256((query or "query").encode("utf-8")).hexdigest()[:8]
        sources = []
        for index in range(limit):
            title, body = _TOPICS[index % len(_TOPICS)]
            slug = f"{digest}-{index}"
            source = self.register(
                f"https://test.invalid/{slug}",
                f"{title} ({slug})",
                body,
                kind="web",
                provider="test-corpus",
                published=f"202{index % 5}-01-01",
                content=body,
            )
            sources.append(source)
        return (
            untrusted_header("test corpus results")
            + f"\nTest corpus results for '{query}' ({len(sources)} hits):\n\n"
            + format_sources(sources)
        )
