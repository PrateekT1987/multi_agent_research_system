"""Search providers.

Mars supports several search backends and degrades gracefully between them:
if an API key is missing (or a provider errors out) the next provider is used.
DuckDuckGo's HTML endpoint is the default because it needs no credentials,
which keeps the system usable out of the box.
"""

from __future__ import annotations

import html
import urllib.parse
from dataclasses import dataclass
from typing import Any, Protocol

from bs4 import BeautifulSoup

from ..config import Settings
from ..domain import scope_query
from ..net import HttpClient, HttpError

__all__ = [
    "SearchResult",
    "SearchProvider",
    "DuckDuckGoProvider",
    "BraveProvider",
    "SerperProvider",
    "TavilyProvider",
    "ExaProvider",
    "build_providers",
    "search",
]

# DuckDuckGo answers bot-identifying agents with an HTTP 202 challenge page
# instead of results, so search requests go out with a plain browser UA.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass(slots=True)
class SearchResult:
    """One hit from a search engine."""

    title: str
    url: str
    snippet: str = ""
    provider: str = "unknown"
    published: str | None = None
    # Filled in by the crawler-governance filter before registration.
    quality: float = 0.0
    license: str = "unknown"
    paywalled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "provider": self.provider,
            "published": self.published,
            "quality": self.quality,
            "license": self.license,
            "paywalled": self.paywalled,
        }


class SearchProvider(Protocol):
    """Protocol implemented by every search backend."""

    name: str

    def is_available(self, settings: Settings) -> bool: ...

    def search(
        self, query: str, max_results: int, settings: Settings, http: HttpClient
    ) -> list[SearchResult]: ...


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(html.unescape(text).split())


class DuckDuckGoProvider:
    """Keyless HTML scraping of DuckDuckGo's result page."""

    name = "duckduckgo"
    endpoint = "https://html.duckduckgo.com/html/"

    def is_available(self, settings: Settings) -> bool:
        return True

    def search(
        self, query: str, max_results: int, settings: Settings, http: HttpClient
    ) -> list[SearchResult]:
        response = http.post(
            self.endpoint,
            data={"q": query, "kl": settings.language},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": BROWSER_USER_AGENT,
            },
        )
        if response.status == 202:
            raise HttpError("duckduckgo returned an anti-bot challenge (HTTP 202)")
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[SearchResult] = []
        for node in soup.select("div.result, div.web-result"):
            link = node.select_one("a.result__a") or node.select_one("a.result-link")
            if link is None:
                continue
            url = self._unwrap(link.get("href", ""))
            title = _clean(link.get_text(" ", strip=True))
            snippet_node = node.select_one("a.result__snippet") or node.select_one(
                "div.result__snippet"
            )
            snippet = _clean(snippet_node.get_text(" ", strip=True)) if snippet_node else ""
            if not url.startswith("http") or not title:
                continue
            results.append(
                SearchResult(title=title, url=url, snippet=snippet, provider=self.name)
            )
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _unwrap(href: str) -> str:
        if href.startswith("//"):
            href = "https:" + href
        if "duckduckgo.com/l/" in href:
            query = urllib.parse.urlparse(href).query
            target = urllib.parse.parse_qs(query).get("uddg")
            if target:
                return target[0]
        return href


class BraveProvider:
    """Brave Search API (needs ``BRAVE_API_KEY``)."""

    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def is_available(self, settings: Settings) -> bool:
        return bool(settings.brave_api_key)

    def search(
        self, query: str, max_results: int, settings: Settings, http: HttpClient
    ) -> list[SearchResult]:
        response = http.get(
            self.endpoint,
            params={"q": query, "count": max_results},
            headers={
                "X-Subscription-Token": settings.brave_api_key or "",
                "Accept": "application/json",
            },
        )
        payload = response.json()
        results: list[SearchResult] = []
        for item in payload.get("web", {}).get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=_clean(item.get("title")),
                    url=item.get("url", ""),
                    snippet=_clean(item.get("description")),
                    provider=self.name,
                    published=item.get("age"),
                )
            )
        return results


class SerperProvider:
    """Serper.dev Google wrapper (needs ``SERPER_API_KEY``)."""

    name = "serper"
    endpoint = "https://google.serper.dev/search"

    def is_available(self, settings: Settings) -> bool:
        return bool(settings.serper_api_key)

    def search(
        self, query: str, max_results: int, settings: Settings, http: HttpClient
    ) -> list[SearchResult]:
        response = http.post(
            self.endpoint,
            json_body={"q": query, "num": max_results},
            headers={
                "X-API-KEY": settings.serper_api_key or "",
                "Content-Type": "application/json",
            },
        )
        payload = response.json()
        results: list[SearchResult] = []
        for item in payload.get("organic", [])[:max_results]:
            results.append(
                SearchResult(
                    title=_clean(item.get("title")),
                    url=item.get("link", ""),
                    snippet=_clean(item.get("snippet")),
                    provider=self.name,
                    published=item.get("date"),
                )
            )
        return results


class TavilyProvider:
    """Tavily search API (needs ``TAVILY_API_KEY``)."""

    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def is_available(self, settings: Settings) -> bool:
        return bool(settings.tavily_api_key)

    def search(
        self, query: str, max_results: int, settings: Settings, http: HttpClient
    ) -> list[SearchResult]:
        response = http.post(
            self.endpoint,
            json_body={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
            },
            headers={"Content-Type": "application/json"},
        )
        payload = response.json()
        results: list[SearchResult] = []
        for item in payload.get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=_clean(item.get("title")),
                    url=item.get("url", ""),
                    snippet=_clean(item.get("content")),
                    provider=self.name,
                    published=item.get("published_date"),
                )
            )
        return results


class ExaProvider:
    """Exa neural search API (needs ``EXA_API_KEY``)."""

    name = "exa"
    endpoint = "https://api.exa.ai/search"

    def is_available(self, settings: Settings) -> bool:
        return bool(settings.exa_api_key)

    def search(
        self, query: str, max_results: int, settings: Settings, http: HttpClient
    ) -> list[SearchResult]:
        response = http.post(
            self.endpoint,
            json_body={
                "apiKey": settings.exa_api_key,
                "query": query,
                "numResults": max_results,
                "contents": {"text": {"maxCharacters": 1200}},
            },
            headers={"Content-Type": "application/json"},
        )
        payload = response.json()
        results: list[SearchResult] = []
        for item in payload.get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=_clean(item.get("title")),
                    url=item.get("url", ""),
                    snippet=_clean(item.get("text"))[:600],
                    provider=self.name,
                    published=item.get("publishedDate"),
                )
            )
        return results


_PROVIDER_CLASSES: dict[str, type] = {
    "duckduckgo": DuckDuckGoProvider,
    "brave": BraveProvider,
    "serper": SerperProvider,
    "tavily": TavilyProvider,
    "exa": ExaProvider,
}


def build_providers(settings: Settings) -> list[SearchProvider]:
    """Instantiate the providers named in ``settings.search_providers``."""
    providers: list[SearchProvider] = []
    for name in settings.search_providers:
        cls = _PROVIDER_CLASSES.get(name.lower())
        if cls is None:
            continue
        provider = cls()
        if provider.is_available(settings):
            providers.append(provider)
    return providers


def search(
    query: str,
    settings: Settings,
    http: HttpClient,
    *,
    max_results: int | None = None,
) -> tuple[list[SearchResult], list[str]]:
    """Run a query across the configured providers.

    Returns ``(results, errors)``; ``errors`` records why a provider failed so
    the agent (and the run log) can see partial degradation.
    """
    limit = max_results or settings.max_search_results
    # Mars is domain-locked: a query without Agentic AI signal is narrowed
    # before it reaches any provider.
    query = scope_query(query)
    errors: list[str] = []
    for provider in build_providers(settings):
        try:
            results = provider.search(query, limit, settings, http)
            if results:
                return results, errors
            errors.append(f"{provider.name}: no results")
        except (HttpError, ValueError, KeyError) as exc:
            errors.append(f"{provider.name}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{provider.name}: unexpected error {exc}")
    return [], errors
