"""Base class for every Mars tool.

The base class gives tools access to the shared :class:`ResearchContext` and
provides the two conventions the rest of the system relies on:

* results are rendered as compact text blocks that are cheap on the context
  window, and
* every retrieved artefact is registered in the evidence store so the agent can
  cite it as ``[S1]``.
"""

from __future__ import annotations

import time
from typing import Any

from crewai.tools import BaseTool
from pydantic import Field

from ..context import ResearchContext
from ..sanitize import wrap_untrusted
from ..schemas import Source

__all__ = ["MarsTool", "format_sources"]


class MarsTool(BaseTool):
    """A tool bound to a research context."""

    context: ResearchContext = Field(default=None, exclude=True)  # type: ignore[assignment]

    # Convenience accessors -------------------------------------------------
    @property
    def settings(self):
        return self.context.settings

    @property
    def evidence(self):
        return self.context.evidence

    @property
    def governance(self):
        return self.context.governance

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Record every tool invocation once, independent of result count."""
        if self.context.watchdog is not None:
            self.context.watchdog.check_run()
        started = time.perf_counter()
        self.evidence.record_tool_call(self.name)
        try:
            result = super().run(*args, **kwargs)
        except Exception as exc:
            if self.context.observer is not None:
                self.context.observer.emit(
                    "tool.call",
                    status="error",
                    tool=self.name,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                    error_type=type(exc).__name__,
                )
            raise
        if self.context.observer is not None:
            self.context.observer.emit(
                "tool.call",
                status="error" if str(result).startswith("Error:") else "ok",
                tool=self.name,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                output_chars=len(str(result)),
            )
        return result

    def register(
        self,
        url: str,
        title: str,
        snippet: str = "",
        *,
        kind: str = "web",
        provider: str = "unknown",
        published: str | None = None,
        content: str | None = None,
        content_type: str = "",
        http_status: int = 0,
        final_url: str = "",
        quality: float = 0.0,
        license: str = "unknown",
        paywalled: bool = False,
        injection_flags: tuple[str, ...] | list[str] = (),
    ) -> Source:
        """Register a source and return it."""
        return self.evidence.register(
            url,
            title,
            snippet,
            kind=kind,
            provider=provider,
            published=published,
            content=content,
            content_type=content_type,
            http_status=http_status,
            final_url=final_url,
            quality=quality,
            license=license,
            paywalled=paywalled,
            injection_flags=injection_flags,
        )

    def untrusted(self, text: str, *, source: str = "", max_chars: int | None = None) -> str:
        """Wrap external text in the untrusted-data boundary."""
        return wrap_untrusted(text, source=source, max_chars=max_chars)

    def governed_register(
        self,
        url: str,
        title: str,
        snippet: str = "",
        *,
        kind: str = "web",
        provider: str = "unknown",
        published: str | None = None,
        content: str | None = None,
        http_status: int = 0,
    ) -> Source | None:
        """Register a source after applying the crawler/governance policy.

        Returns ``None`` when the policy rejects the destination or its content.
        """
        decision = (
            self.governance.check_content(
                url,
                "\n".join(part for part in (title, snippet, content or "") if part),
                published=published,
            )
            if self.governance is not None
            else None
        )
        if decision is not None and not decision.allowed:
            return None
        cap = getattr(self.context.settings, "max_sources_per_domain", 0)
        if self.evidence.domain_cap_reached(url, cap):
            return None
        return self.register(
            url,
            title,
            snippet,
            kind=kind,
            provider=provider,
            published=published,
            content=content,
            http_status=http_status,
            quality=decision.quality if decision else 0.0,
            license=decision.license if decision else "unknown",
            paywalled=decision.paywalled if decision else False,
        )


def format_sources(sources: list[Source], *, max_snippet: int = 320) -> str:
    """Render sources as a citation-ready text block."""
    if not sources:
        return "No results found."
    lines: list[str] = []
    for source in sources:
        snippet = source.snippet.strip()
        if len(snippet) > max_snippet:
            snippet = snippet[:max_snippet].rstrip() + "..."
        lines.append(f"[{source.id}] {source.title}")
        lines.append(f"    URL: {source.url}")
        if snippet:
            lines.append(f"    Snippet: {snippet}")
        if source.published:
            lines.append(f"    Published: {source.published}")
        lines.append("")
    return "\n".join(lines).strip()


def truncate(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` characters, cutting on a word boundary."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut[-200:]:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip() + "\n\n[... truncated ...]"
