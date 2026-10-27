"""Web search and page retrieval tools."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from ..domain import scope_query
from ..governance import filter_search_results
from ..net import HttpError
from ..sanitize import sanitize_untrusted, untrusted_header
from .base import MarsTool, format_sources, truncate
from .search import search

__all__ = ["WebSearchTool", "WebFetchTool", "SearchArgs", "FetchArgs"]

_SCRIPT_STYLE = re.compile(r"<(script|style|noscript|svg|iframe)\b.*?</\1>", re.S | re.I)
_TAGS = re.compile(r"<[^>]+>")


class SearchArgs(BaseModel):
    """Arguments for :class:`WebSearchTool`."""

    query: str = Field(description="Search query. Use specific phrases, names, or quoted terms.")
    max_results: int = Field(default=8, description="How many results to return (1-20).")


class FetchArgs(BaseModel):
    """Arguments for :class:`WebFetchTool`."""

    url: str = Field(description="Absolute http(s) URL of the page to read.")
    max_chars: int = Field(default=10000, description="Maximum characters of page text to return.")


class WebSearchTool(MarsTool):
    """Search the web and return ranked results with stable citation ids.

    Use this to discover sources. Results are returned as ``[S1] Title / URL /
    Snippet`` blocks - cite them by id when you write up findings.
    """

    name: str = "web_search"
    description: str = (
        "Search the web for information about a topic. Returns up to 10 ranked results "
        "each with a citation id like [S1] that you should use when citing sources."
    )
    args_schema: type[BaseModel] = SearchArgs

    def _run(self, query: str = "", max_results: int = 8, **_: Any) -> str:
        if not query.strip():
            return "Error: query must not be empty."
        query = scope_query(query)
        limit = max(1, min(int(max_results or 8), 20))
        results, errors = search(
            query, self.context.settings, self.context.http, max_results=limit
        )
        if not results:
            hint = (" Provider errors: " + "; ".join(errors)) if errors else ""
            return f"No results for '{query}'.{hint}"

        accepted, rejected = filter_search_results(
            results, self.context.settings, http=self.context.http
        )
        if not accepted:
            reasons = "; ".join(reason for _, reason in rejected[:3])
            return (
                f"No results for '{query}' passed the crawler policy."
                + (f" Rejected: {reasons}" if reasons else "")
            )

        cap = getattr(self.context.settings, "max_sources_per_domain", 0)
        sources = []
        capped = 0
        for item in accepted:
            if self.evidence.domain_cap_reached(item.url, cap):
                capped += 1
                continue
            sources.append(
                self.register(
                    item.url,
                    sanitize_untrusted(item.title).text,
                    sanitize_untrusted(item.snippet).text,
                    kind="web",
                    provider=item.provider,
                    published=item.published,
                    quality=getattr(item, "quality", 0.0),
                    license=getattr(item, "license", "unknown"),
                    paywalled=getattr(item, "paywalled", False),
                )
            )
        header = f"Search results for '{query}' ({len(sources)} hits"
        if rejected:
            header += f", {len(rejected)} rejected by policy"
        if capped:
            header += f", {capped} over the per-domain cap"
        header += "):"
        return f"{untrusted_header('search results')}\n{header}\n\n" + format_sources(sources)


class WebFetchTool(MarsTool):
    """Fetch a web page and return its readable text.

    Use this after :class:`WebSearchTool` to read the promising sources. The
    page is registered in the evidence store, so cite it with its id.
    """

    name: str = "fetch_page"
    description: str = (
        "Download a web page and return its main text content. Use after web_search "
        "to read a specific source. The page is registered as a citable source."
    )
    args_schema: type[BaseModel] = FetchArgs

    def _run(self, url: str = "", max_chars: int = 10000, **_: Any) -> str:
        if not url.startswith(("http://", "https://")):
            return "Error: url must start with http:// or https://"
        decision = self.governance.check_fetch(url) if self.governance else None
        if decision is not None and not decision.allowed:
            return f"Error: {url} is not permitted by the crawler policy ({decision.reason})."
        limit = max(500, min(int(max_chars or 10000), 60000))
        try:
            response = self.context.http.get(
                url,
                allowed_content_types={
                    "text/html",
                    "text/plain",
                    "application/xhtml+xml",
                    "application/xml",
                    "text/xml",
                    "application/json",
                },
            )
        except HttpError as exc:
            return f"Error: could not fetch {url} ({exc})."

        raw_text = extract_readable_text(response.text)
        if not raw_text:
            return f"Error: no readable text found at {url}."

        content_decision = (
            self.governance.check_content(
                str(response.url),
                f"{page_title(response.text)}\n{raw_text}",
            )
            if self.governance
            else None
        )
        if content_decision is not None and not content_decision.allowed:
            return (
                f"Error: {response.url} is not permitted by the crawler policy "
                f"({content_decision.reason})."
            )

        cap = getattr(self.context.settings, "max_sources_per_domain", 0)
        if self.evidence.domain_cap_reached(str(response.url), cap):
            return (
                f"Error: the per-domain source cap ({cap}) is already reached for "
                f"{response.url}; this page was not registered. Cite the sources "
                "already collected for this domain."
            )

        sanitized = sanitize_untrusted(raw_text)
        source = self.register(
            str(response.url),
            sanitize_untrusted(page_title(response.text) or str(response.url)).text,
            raw_text[:400],
            kind="web",
            provider="fetch",
            content=raw_text,
            content_type=response.headers.get("content-type", ""),
            http_status=response.status,
            final_url=str(response.url),
            quality=content_decision.quality if content_decision else 0.0,
            license=content_decision.license if content_decision else "unknown",
            paywalled=content_decision.paywalled if content_decision else False,
            injection_flags=tuple(sanitized.flags),
        )
        body = truncate(sanitized.text, limit)
        flags = (
            f"\nInjection patterns neutralised: {', '.join(sanitized.flags)}"
            if sanitized.flags
            else ""
        )
        return (
            f"[{source.id}] {source.title}\n"
            f"URL: {source.url}\n"
            f"Characters: {len(raw_text)}\n"
            f"Content hash: {source.content_hash or 'not archived'}{flags}\n\n"
            f"{self.untrusted(body, source=source.id)}"
        )


def page_title(html: str) -> str:
    """Extract the ``<title>`` of an HTML document."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.S | re.I)
    if not match:
        return ""
    return " ".join(match.group(1).split())[:200]


def extract_readable_text(html: str) -> str:
    """Turn an HTML document into readable plain text.

    A full readability implementation is overkill here: we drop scripts, styles
    and navigation, then collapse whitespace while keeping paragraph breaks.
    """
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "iframe", "form", "nav", "footer"]):
            tag.decompose()
        for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
            tag.insert_before("\n\n## ")
            tag.insert_after("\n")
        for tag in soup.find_all(["p", "li", "tr", "blockquote", "pre"]):
            tag.insert_after("\n")
        text = soup.get_text("\n")
    except Exception:  # pragma: no cover - fall back to regex stripping
        text = _TAGS.sub(" ", _SCRIPT_STYLE.sub(" ", html))

    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()
