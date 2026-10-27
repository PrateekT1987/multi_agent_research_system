"""Release metadata for the prompt bundle.

This module is deliberately dependency-free so configuration can validate
prompt versions without importing the agent package (which imports the
pipeline).  The prompt text itself lives in :mod:`mars.agents.profiles` and
:mod:`mars.agents.tasks`; this file records which instruction set those files
represent.
"""

from __future__ import annotations

__all__ = [
    "PROMPT_VERSION",
    "PROMPT_CHANNEL",
    "PROMPT_RELEASED_AT",
    "PROMPT_CHANGELOG",
    "PROMPT_CHANNELS",
]

# Bump this whenever the prompt text changes materially.  It is written into
# every report and run manifest so an artefact can be tied to the exact
# instruction set that produced it.
PROMPT_VERSION = "2026-10-27.p4-prompt-registry"

# The channel this built-in bundle belongs to.  Operators can select a
# different channel with MARS_PROMPT_CHANNEL; non-production channels are
# surfaced as warnings at startup.
PROMPT_CHANNEL = "prod"

PROMPT_RELEASED_AT = "2026-10-27"

PROMPT_CHANGELOG: tuple[str, ...] = (
    "2026-10-27.p4-prompt-registry: central citation protocol reused by the "
    "writer and gap researcher, versioned bundles, channels and override files.",
    "2026-10-26.p3-agentic-ai-domain: fixed Agentic AI taxonomy, domain "
    "protocol, topic gate and retrieval scoping.",
    "2026-10-26.p2: production hardening (untrusted-data protocol, citation "
    "discipline, gap reporting).",
)

PROMPT_CHANNELS: tuple[str, ...] = ("dev", "staging", "prod")
