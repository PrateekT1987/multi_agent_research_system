"""Versioned prompt bundle and hot-swappable overrides.

The prompt text is code-reviewed in ``profiles.py`` and ``tasks.py``, but
production teams also need to:

* tie every run to an exact prompt version (recorded in the manifest),
* tag a bundle for ``dev`` / ``staging`` / ``prod``,
* roll back a safety-critical protocol block without rebuilding the image.

The built-in bundle is always available.  An optional JSON override file
(``MARS_PROMPT_OVERRIDES``) can replace the citation, untrusted-data or domain
protocol text and append agent-specific instructions.  The override file is
versioned in git; rolling back is a matter of pointing the environment at the
previous file and restarting, with no code deploy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from ..prompt_meta import (
    PROMPT_CHANGELOG,
    PROMPT_CHANNEL,
    PROMPT_CHANNELS,
    PROMPT_RELEASED_AT,
    PROMPT_VERSION,
)
from .profiles import (
    CITATION_PROTOCOL,
    DOMAIN_PROTOCOL,
    PROFILES,
    UNTRUSTED_DATA_PROTOCOL,
)

__all__ = [
    "BUILTIN_PROMPTS",
    "PromptBundle",
    "PromptRegistryError",
    "active_prompts",
    "configure_prompts",
    "configure_prompts_from_settings",
    "load_prompt_bundle",
    "reset_prompts",
]

PromptChannel = Literal["dev", "staging", "prod"]


class PromptRegistryError(ValueError):
    """The selected prompt version, channel or override file is invalid."""


@dataclass(frozen=True, slots=True)
class PromptBundle:
    """One immutable prompt instruction set."""

    version: str = PROMPT_VERSION
    channel: str = PROMPT_CHANNEL
    released_at: str = PROMPT_RELEASED_AT
    changelog: tuple[str, ...] = PROMPT_CHANGELOG
    citation_protocol: str = CITATION_PROTOCOL
    untrusted_protocol: str = UNTRUSTED_DATA_PROTOCOL
    domain_protocol: str = DOMAIN_PROTOCOL
    agent_suffixes: dict[str, str] = field(default_factory=dict)
    source: str = "builtin"

    def checksum(self) -> str:
        """Stable digest of the effective instruction text."""
        payload = {
            "version": self.version,
            "channel": self.channel,
            "citation_protocol": self.citation_protocol,
            "untrusted_protocol": self.untrusted_protocol,
            "domain_protocol": self.domain_protocol,
            "agent_suffixes": dict(sorted(self.agent_suffixes.items())),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def manifest(self) -> dict[str, Any]:
        """Serialise the bundle for a report or run manifest (text omitted)."""
        return {
            "version": self.version,
            "channel": self.channel,
            "released_at": self.released_at,
            "checksum": self.checksum(),
            "source": self.source,
            "changelog": list(self.changelog),
            "agent_suffixes": sorted(self.agent_suffixes),
        }


BUILTIN_PROMPTS = PromptBundle()

_ACTIVE: PromptBundle = BUILTIN_PROMPTS


def active_prompts() -> PromptBundle:
    """Return the prompt bundle in force for this process."""
    return _ACTIVE


def configure_prompts(bundle: PromptBundle) -> PromptBundle:
    """Install ``bundle`` as the active instruction set."""
    global _ACTIVE
    _ACTIVE = bundle
    return bundle


def reset_prompts() -> None:
    """Restore the built-in bundle (used by tests and library callers)."""
    global _ACTIVE
    _ACTIVE = BUILTIN_PROMPTS


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptRegistryError(f"cannot read prompt overrides {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PromptRegistryError(f"prompt overrides {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PromptRegistryError(f"prompt overrides {path} must contain a JSON object")
    return data


def load_prompt_bundle(
    path: str | Path | None = None,
    *,
    version: str | None = None,
    channel: str | None = None,
) -> PromptBundle:
    """Build the effective bundle from the built-in text plus optional overrides.

    ``version`` and ``channel`` are the operator's requested values.  An
    override file may carry its own values; if both are supplied they must
    agree, so a stale override cannot masquerade as a different release.
    """
    data: dict[str, Any] = {}
    source = "builtin"
    if path:
        override_path = Path(path).expanduser()
        data = _load_json(override_path)
        source = str(override_path)

    requested_version = version or data.get("prompt_version") or PROMPT_VERSION
    requested_channel = channel or data.get("channel") or PROMPT_CHANNEL

    file_version = data.get("prompt_version")
    if version and file_version and version != file_version:
        raise PromptRegistryError(
            f"MARS_PROMPT_VERSION={version!r} does not match the override file "
            f"version {file_version!r}"
        )
    file_channel = data.get("channel")
    if channel and file_channel and channel != file_channel:
        raise PromptRegistryError(
            f"MARS_PROMPT_CHANNEL={channel!r} does not match the override file "
            f"channel {file_channel!r}"
        )

    if not isinstance(requested_version, str) or not requested_version.strip():
        raise PromptRegistryError("prompt version must be a non-empty string")
    if requested_channel not in PROMPT_CHANNELS:
        raise PromptRegistryError(
            f"unknown prompt channel {requested_channel!r}; "
            f"expected one of {', '.join(PROMPT_CHANNELS)}"
        )
    # Without an override file the only reproducible text is the built-in one.
    if requested_version != PROMPT_VERSION and not path:
        raise PromptRegistryError(
            f"prompt version {requested_version!r} is not built in; supply "
            "MARS_PROMPT_OVERRIDES with that version's text"
        )

    protocols = data.get("protocols") or {}
    if not isinstance(protocols, dict):
        raise PromptRegistryError("prompt overrides 'protocols' must be an object")
    citation = protocols.get("citation", CITATION_PROTOCOL)
    untrusted = protocols.get("untrusted", UNTRUSTED_DATA_PROTOCOL)
    domain = protocols.get("domain", DOMAIN_PROTOCOL)
    for name, value in (
        ("citation", citation),
        ("untrusted", untrusted),
        ("domain", domain),
    ):
        if not isinstance(value, str) or not value.strip():
            raise PromptRegistryError(f"prompt protocol {name!r} must be non-empty text")

    suffixes = data.get("agent_suffixes") or {}
    if not isinstance(suffixes, dict):
        raise PromptRegistryError("prompt overrides 'agent_suffixes' must be an object")
    unknown = sorted(set(suffixes) - set(PROFILES))
    if unknown:
        raise PromptRegistryError(
            f"prompt overrides reference unknown agent(s): {', '.join(unknown)}"
        )
    clean_suffixes: dict[str, str] = {}
    for key, value in suffixes.items():
        if not isinstance(value, str) or not value.strip():
            raise PromptRegistryError(f"agent suffix for {key!r} must be non-empty text")
        clean_suffixes[str(key)] = value.strip()

    changelog = data.get("changelog") or PROMPT_CHANGELOG
    if not isinstance(changelog, (list, tuple)):
        raise PromptRegistryError("prompt overrides 'changelog' must be a list of strings")

    return replace(
        BUILTIN_PROMPTS,
        version=requested_version,
        channel=requested_channel,
        changelog=tuple(str(item) for item in changelog),
        citation_protocol=citation,
        untrusted_protocol=untrusted,
        domain_protocol=domain,
        agent_suffixes=clean_suffixes,
        source=source,
    )


def configure_prompts_from_settings(settings: Any) -> PromptBundle:
    """Load and install the bundle selected by a ``Settings`` object."""
    bundle = load_prompt_bundle(
        getattr(settings, "prompt_overrides_path", None),
        version=getattr(settings, "prompt_version", None),
        channel=getattr(settings, "prompt_channel", None),
    )
    return configure_prompts(bundle)
