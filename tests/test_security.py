"""Security regression tests for the P0 hardening work."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mars.config import REDACTED, Settings
from mars.context import ResearchContext
from mars.net import HttpClient, HttpError, validate_url
from mars.tools import NETWORK_TOOL_NAMES, build_toolset


# --------------------------------------------------------------------- secrets
def test_to_dict_redacts_secrets_by_default() -> None:
    settings = Settings(
        openrouter_api_key="sk-live-secret",
        tavily_api_key="tvly-secret",
        approval_secret="approval-secret",
    )

    data = settings.to_dict()

    assert data["openrouter_api_key"] == REDACTED
    assert data["tavily_api_key"] == REDACTED
    assert data["approval_secret"] == REDACTED
    assert "sk-live-secret" not in json.dumps(data)


def test_to_dict_can_return_raw_values_for_in_memory_use() -> None:
    settings = Settings(openrouter_api_key="sk-live-secret")

    assert settings.to_dict(redact_secrets=False)["openrouter_api_key"] == "sk-live-secret"


def test_to_dict_redacts_secret_like_extra_values() -> None:
    settings = Settings(extra={"huggingface_token": "hf_secret", "label": "keep"})

    data = settings.to_dict()

    assert data["extra"]["huggingface_token"] == REDACTED
    assert data["extra"]["label"] == "keep"


# ------------------------------------------------------------------------ SSRF
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file",
        "http://user:password@example.com/",
        "http://localhost/",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://metadata.google.internal/",
        "http://printer.local/",
    ],
)
def test_validate_url_refuses_unsafe_destinations(url: str) -> None:
    with pytest.raises(HttpError):
        validate_url(url)


def test_validate_url_accepts_a_public_https_url() -> None:
    validate_url("https://example.com/article")


def test_validate_url_allows_loopback_only_when_explicitly_enabled() -> None:
    validate_url("http://127.0.0.1:8000/", allow_private_network=True)


# ---------------------------------------------------------------- offline mode
def test_offline_client_refuses_to_make_requests() -> None:
    client = HttpClient(allow_network=False)
    try:
        with pytest.raises(HttpError, match="network access is disabled"):
            client.get("https://example.com")
    finally:
        client.close()


def test_offline_toolset_withholds_every_network_tool(settings) -> None:
    offline = settings.with_overrides(allow_network=False)
    ctx = ResearchContext(settings=offline)
    try:
        names = {tool.name for tool in build_toolset(ctx, track="history")}
    finally:
        ctx.close()

    assert not (names & NETWORK_TOOL_NAMES)
    assert {"record_finding", "list_sources", "read_findings"} <= names
    # Production ships no curated corpus.
    assert "query_corpus" not in names


# ---------------------------------------------------------------- HTTP limits
class _PayloadHandler(BaseHTTPRequestHandler):
    body = b"{}"
    content_type = "application/json"
    location: str | None = None

    def log_message(self, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.location:
            self.send_response(302)
            self.send_header("Location", self.location)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", self.content_type)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)


@pytest.fixture
def payload_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _PayloadHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_response_size_is_capped(payload_server) -> None:
    _PayloadHandler.body = b"x" * 5000
    _PayloadHandler.location = None
    client = HttpClient(
        allow_private_network=True,
        max_response_bytes=1024,
        cache_enabled=False,
        min_seconds_between_requests=0,
    )
    try:
        with pytest.raises(HttpError, match="exceeded"):
            client.get(f"{payload_server}/big")
    finally:
        client.close()


def test_redirects_are_revalidated(payload_server) -> None:
    _PayloadHandler.location = "file:///etc/passwd"
    client = HttpClient(
        allow_private_network=True,
        cache_enabled=False,
        min_seconds_between_requests=0,
    )
    try:
        with pytest.raises(HttpError, match="non-HTTP"):
            client.get(f"{payload_server}/redirect")
    finally:
        client.close()


def test_unexpected_content_types_are_refused(payload_server) -> None:
    _PayloadHandler.location = None
    _PayloadHandler.body = b"\x00\x01binary"
    _PayloadHandler.content_type = "application/octet-stream"
    client = HttpClient(
        allow_private_network=True,
        cache_enabled=False,
        min_seconds_between_requests=0,
    )
    try:
        with pytest.raises(HttpError, match="content type"):
            client.get(
                f"{payload_server}/binary",
                allowed_content_types={"text/html", "text/plain"},
            )
    finally:
        client.close()


# --------------------------------------------------------- untrusted rendering
def test_streamlit_never_renders_findings_as_raw_html() -> None:
    source = (Path(__file__).parents[1] / "src" / "mars" / "app.py").read_text(
        encoding="utf-8"
    )

    assert "unsafe_allow_html=True" not in source
    assert "unsafe_allow_html" not in source


def test_production_package_has_no_offline_or_corpus_paths() -> None:
    """The curated corpus, mock provider and OpenCode path are test-only."""
    root = Path(__file__).parents[1] / "src" / "mars"
    forbidden = (
        "MockLLM",
        "enable_curated_corpus",
        "query_corpus",
        "opencode",
        "opencode_api_key",
    )
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{needle!r} leaked into {path}"
