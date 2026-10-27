"""Academic-paper and reference-encyclopedia tools."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from pydantic import BaseModel, Field

from ..domain import is_domain_topic, scope_query
from ..net import HttpError
from ..sanitize import sanitize_untrusted, untrusted_header
from .base import MarsTool, format_sources, truncate

__all__ = [
    "ArxivSearchTool",
    "SemanticScholarTool",
    "WikipediaTool",
    "HackerNewsTool",
    "ArxivArgs",
    "SemanticScholarArgs",
    "WikipediaArgs",
    "HackerNewsArgs",
]

_ATOM = "{http://www.w3.org/2005/Atom}"


class ArxivArgs(BaseModel):
    query: str = Field(description="Search terms, e.g. 'multi-agent LLM orchestration'.")
    max_results: int = Field(default=8, description="Number of papers to return (1-25).")


class SemanticScholarArgs(BaseModel):
    query: str = Field(description="Paper search terms.")
    max_results: int = Field(default=6, description="Number of papers to return (1-15).")


class WikipediaArgs(BaseModel):
    query: str = Field(description="Article title or search term.")


class HackerNewsArgs(BaseModel):
    query: str = Field(description="Search terms for Hacker News stories.")
    max_results: int = Field(default=6, description="Number of stories to return (1-20).")


class ArxivSearchTool(MarsTool):
    """Search arXiv for academic papers.

    Best for: technical foundations, evaluation results, and the research
    literature behind a topic. Returns papers with citation ids.
    """

    name: str = "arxiv_search"
    description: str = (
        "Search arXiv for academic papers on a topic. Returns titles, dates, "
        "abstracts and links, each with a citation id such as [S1]."
    )
    args_schema: type[BaseModel] = ArxivArgs

    def _run(self, query: str = "", max_results: int = 8, **_: Any) -> str:
        if not query.strip():
            return "Error: query must not be empty."
        query = scope_query(query)
        limit = max(1, min(int(max_results or 8), 25))
        url = "https://export.arxiv.org/api/query"
        try:
            response = self.context.http.get(
                url,
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": limit,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
            )
        except HttpError as exc:
            return f"Error: arXiv request failed ({exc})."

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            return f"Error: could not parse arXiv response ({exc})."

        entries = root.findall(f"{_ATOM}entry")
        if not entries:
            return f"No arXiv papers found for '{query}'."

        sources = []
        for entry in entries:
            title = _text(entry, f"{_ATOM}title")
            link = _text(entry, f"{_ATOM}id")
            summary = _text(entry, f"{_ATOM}summary")
            published = _text(entry, f"{_ATOM}published")[:10]
            authors = [
                _text(author, f"{_ATOM}name")
                for author in entry.findall(f"{_ATOM}author")
            ]
            if not title or not link:
                continue
            snippet = summary
            if authors:
                snippet = f"{', '.join(authors[:4])} - {summary}"
            source = self.governed_register(
                link,
                sanitize_untrusted(title).text,
                sanitize_untrusted(snippet).text,
                kind="paper",
                provider="arxiv",
                published=published,
                content=summary,
            )
            if source is not None:
                sources.append(source)

        header = f"arXiv papers for '{query}' ({len(sources)} hits):"
        return untrusted_header("arxiv results") + "\n" + header + "\n\n" + format_sources(sources)


class SemanticScholarTool(MarsTool):
    """Search Semantic Scholar for papers (optional, rate limited)."""

    name: str = "semantic_scholar_search"
    description: str = (
        "Search Semantic Scholar's academic graph for papers, citation counts "
        "and venues. Useful for finding influential work on a topic."
    )
    args_schema: type[BaseModel] = SemanticScholarArgs

    def _run(self, query: str = "", max_results: int = 6, **_: Any) -> str:
        if not query.strip():
            return "Error: query must not be empty."
        query = scope_query(query)
        limit = max(1, min(int(max_results or 6), 15))
        try:
            response = self.context.http.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "title,abstract,year,url,venue,citationCount,authors",
                },
            )
        except HttpError as exc:
            return f"Error: Semantic Scholar request failed ({exc}). Rate limits are common; carry on."

        try:
            payload = response.json()
        except ValueError:
            return "Error: unexpected response from Semantic Scholar."

        papers = payload.get("data") or []
        if not papers:
            return f"No papers found for '{query}'."

        sources = []
        for paper in papers:
            title = paper.get("title") or ""
            url = paper.get("url") or ""
            if not title or not url:
                continue
            authors = ", ".join(a.get("name", "") for a in (paper.get("authors") or [])[:4])
            snippet = (
                f"{authors}. {paper.get('venue') or 'n/a'} ({paper.get('year') or 'n/a'}). "
                f"Citations: {paper.get('citationCount', 0)}. {paper.get('abstract') or ''}"
            )
            source = self.governed_register(
                url,
                sanitize_untrusted(title).text,
                sanitize_untrusted(snippet).text,
                kind="paper",
                provider="semantic_scholar",
                published=str(paper.get("year") or "") or None,
                content=paper.get("abstract") or snippet,
            )
            if source is not None:
                sources.append(source)
        return (
            untrusted_header("semantic scholar results")
            + "\n"
            + f"Semantic Scholar papers for '{query}' ({len(sources)} hits):\n\n"
            + format_sources(sources)
        )


class WikipediaTool(MarsTool):
    """Look up encyclopedia summaries for background and definitions."""

    name: str = "wikipedia_lookup"
    description: str = (
        "Look up an article on Wikipedia for a background summary, definition, "
        "or historical context. Returns the extract plus the page URL."
    )
    args_schema: type[BaseModel] = WikipediaArgs

    def _run(self, query: str = "", **_: Any) -> str:
        if not query.strip():
            return "Error: query must not be empty."
        if not is_domain_topic(query):
            return (
                "Error: Mars only researches Agentic AI; "
                f"{query!r} is outside the domain. Use an agentic-AI term instead."
            )
        title = query.strip().replace(" ", "_")
        try:
            response = self.context.http.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
            )
        except HttpError as exc:
            return f"Error: Wikipedia request failed ({exc})."

        if response.status == 404:
            return f"No Wikipedia article found for '{query}'. Try a different term."

        try:
            payload = response.json()
        except ValueError:
            return "Error: unexpected response from Wikipedia."

        page_title = payload.get("title") or query
        extract = payload.get("extract") or ""
        page_url = payload.get("content_urls", {}).get("desktop", {}).get("page") or (
            f"https://en.wikipedia.org/wiki/{title}"
        )
        if not extract:
            return f"No summary available for '{query}'."

        sanitized = sanitize_untrusted(extract)
        source = self.governed_register(
            page_url,
            f"Wikipedia: {sanitize_untrusted(page_title).text}",
            sanitized.text[:400],
            kind="wiki",
            provider="wikipedia",
            content=extract,
        )
        if source is None:
            return f"Error: {page_url} is not permitted by the crawler policy."
        return (
            f"[{source.id}] {source.title}\n"
            f"URL: {source.url}\n\n"
            f"{self.untrusted(truncate(sanitized.text, 4000), source=source.id)}"
        )


class HackerNewsTool(MarsTool):
    """Search Hacker News for practitioner discussion."""

    name: str = "hackernews_search"
    description: str = (
        "Search Hacker News for practitioner discussion and links about a topic. "
        "Useful for gauging what practitioners actually use and what they worry about."
    )
    args_schema: type[BaseModel] = HackerNewsArgs

    def _run(self, query: str = "", max_results: int = 6, **_: Any) -> str:
        if not query.strip():
            return "Error: query must not be empty."
        query = scope_query(query)
        limit = max(1, min(int(max_results or 6), 20))
        try:
            response = self.context.http.get(
                "https://hn.algolia.com/api/v1/search",
                params={"query": query, "hitsPerPage": limit, "tags": "story"},
            )
        except HttpError as exc:
            return f"Error: Hacker News request failed ({exc})."

        try:
            payload = response.json()
        except ValueError:
            return "Error: unexpected response from Hacker News."

        hits = payload.get("hits") or []
        if not hits:
            return f"No Hacker News stories found for '{query}'."

        sources = []
        for hit in hits:
            title = hit.get("title") or hit.get("story_title") or ""
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            if not title:
                continue
            points = hit.get("points") or 0
            comments = hit.get("num_comments") or 0
            snippet = f"{points} points, {comments} comments. {hit.get('story_text') or ''}"
            source = self.governed_register(
                url,
                sanitize_untrusted(title).text,
                sanitize_untrusted(snippet).text,
                kind="community",
                provider="hackernews",
                published=hit.get("created_at"),
                content=hit.get("story_text") or snippet,
            )
            if source is not None:
                sources.append(source)
        return (
            untrusted_header("hacker news results")
            + "\n"
            + f"Hacker News stories for '{query}' ({len(sources)} hits):\n\n"
            + format_sources(sources)
        )


def _text(node: ET.Element, path: str) -> str:
    found = node.find(path)
    if found is None or found.text is None:
        return ""
    return " ".join(found.text.split())
