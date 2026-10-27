"""Tests for the retrieval tools."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from mars.config import build_settings
from mars.context import ResearchContext
from mars.tools import build_toolset
from mars.tools.web import WebFetchTool, WebSearchTool, extract_readable_text

HTML_PAGE = """
<html>
  <head><title>Agentic AI explained</title></head>
  <body>
    <nav><a href="/">Home</a></nav>
    <script>console.log('nope')</script>
    <h1>Agentic AI</h1>
    <p>Agents use tools to accomplish goals beyond a single completion.</p>
    <ul><li>They plan.</li><li>They act.</li></ul>
    <footer>footer text</footer>
  </body>
</html>
"""


class _SearchHandler(BaseHTTPRequestHandler):
    """Serves a fake results page for POST and a page for GET."""

    status = 200

    def log_message(self, *args: object) -> None:
        return

    def do_GET(self) -> None:
        payload = HTML_PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = (
            b'<div class="result"><a class="result__a" href="https://example.com/a">'
            b"Example A</a><a class=\"result__snippet\">LLM agents use tools to plan and act</a></div>"
            b'<div class="result"><a class="result__a" href="https://example.com/b">'
            b"Example B</a></div>"
        )
        self.send_response(self.status)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def search_site():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _SearchHandler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_extract_readable_text_drops_noise() -> None:
    text = extract_readable_text(HTML_PAGE)
    assert "Agents use tools" in text
    assert "console.log" not in text
    assert "footer text" not in text


def _no_key_settings():
    # Use empty strings to override env vars (with_overrides ignores None)
    # Include duckduckgo in search_providers since keyed providers are disabled
    return build_settings(
        topic="agentic AI",
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        cache_enabled=False,
        tavily_api_key="",
        brave_api_key="",
        serper_api_key="",
        exa_api_key="",
        search_providers=("duckduckgo",),
        allow_network=True,
        allow_private_network=True,
        env_file=None,
    )


def test_web_search_registers_citable_sources(search_site) -> None:
    settings = _no_key_settings()
    ctx = ResearchContext(settings=settings)
    try:
        tool = WebSearchTool(context=ctx)
        from mars.tools import search as search_module

        original = search_module.DuckDuckGoProvider.endpoint
        search_module.DuckDuckGoProvider.endpoint = f"{search_site}/search"
        try:
            output = tool._run(query="agentic ai", max_results=5)
        finally:
            search_module.DuckDuckGoProvider.endpoint = original

        assert "[S1]" in output
        assert "Example A" in output
        sources = ctx.evidence.sources()
        assert sources[0].url == "https://example.com/a"
        assert sources[0].kind == "web"
    finally:
        ctx.close()


def test_web_search_reports_an_anti_bot_challenge(search_site) -> None:
    """A 202 challenge page must be reported, not silently returned as 'no results'."""
    settings = _no_key_settings()
    ctx = ResearchContext(settings=settings)
    try:
        tool = WebSearchTool(context=ctx)
        from mars.tools import search as search_module

        original_endpoint = search_module.DuckDuckGoProvider.endpoint
        original_status = _SearchHandler.status
        search_module.DuckDuckGoProvider.endpoint = f"{search_site}/search"
        _SearchHandler.status = 202
        try:
            output = tool._run(query="agentic ai", max_results=5)
        finally:
            search_module.DuckDuckGoProvider.endpoint = original_endpoint
            _SearchHandler.status = original_status

        assert "No results" in output
        assert "anti-bot" in output
        assert ctx.evidence.sources() == []
    finally:
        ctx.close()


def test_web_fetch_registers_the_page(search_site) -> None:
    settings = _no_key_settings()
    ctx = ResearchContext(settings=settings)
    try:
        tool = WebFetchTool(context=ctx)
        output = tool._run(url=f"{search_site}/page")

        assert "Agentic AI explained" in output
        assert "Agents use tools" in output
        assert ctx.evidence.sources()[0].title == "Agentic AI explained"
    finally:
        ctx.close()


def test_web_fetch_rejects_bad_urls() -> None:
    settings = _no_key_settings()
    ctx = ResearchContext(settings=settings)
    try:
        tool = WebFetchTool(context=ctx)
        assert tool._run(url="not-a-url").startswith("Error")
    finally:
        ctx.close()


def test_toolset_contains_expected_tools(network_context) -> None:
    tools = build_toolset(network_context, track="history")
    names = {tool.name for tool in tools}
    assert {
        "web_search",
        "fetch_page",
        "arxiv_search",
        "wikipedia_lookup",
        "record_finding",
        "list_sources",
        "read_findings",
    } <= names
    assert "query_corpus" not in names


def test_record_finding_flags_unknown_sources(context) -> None:
    tools = {tool.name: tool for tool in build_toolset(context, track="history")}
    output = tools["record_finding"]._run(
        claim="Agents plan.", source_ids="S99", confidence=0.5
    )
    assert "S99" in output
    assert "not registered" in output
