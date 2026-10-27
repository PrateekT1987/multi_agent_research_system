"""Shared pytest fixtures.

The most interesting fixture here is :func:`fake_openrouter`: a local HTTP
server that speaks the OpenRouter/OpenAI chat-completions protocol.  It lets
the test-suite exercise the *real* client stack (langchain-openrouter -> the
OpenRouter SDK -> HTTP -> CrewAI's tool-calling loop) without needing an API
key or any network access.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from mars.config import Settings, build_settings
from mars.context import ResearchContext

CITATION_RE = re.compile(r"\[(S\d+)\]")

#: Environment prefixes that a real deployment or developer shell may export.
#: They are scrubbed for every test so the suite never depends on the host.
_AMBIENT_ENV_PREFIXES = (
    "MARS_",
    "OPENROUTER_",
    "TAVILY_",
    "BRAVE_",
    "SERPER_",
    "EXA_",
)


class _FakeState:
    """Mutable state shared with the request handler."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.tool_result_ids: list[str] = []


def _openrouter_response(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None):
    """Build an OpenRouter-style response for the given conversation."""
    from tests.support.fake_llm import MockLLM

    has_tool_result = any(message.get("role") == "tool" for message in messages)
    tool_names = [
        (tool.get("function") or {}).get("name")
        for tool in (tools or [])
        if isinstance(tool, dict)
    ]

    if tools and not has_tool_result:
        # Ask for a corpus query - it works offline and registers real sources.
        target = "query_corpus" if "query_corpus" in tool_names else tool_names[0]
        arguments = {"query": "agentic AI history current future"}
        if target == "web_search":
            arguments = {"query": "agentic AI"}
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_test_1",
                    "type": "function",
                    "function": {
                        "name": target,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }

    system_text = "\n".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "system"
    )
    user_text = "\n".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "user"
    )
    tool_text = "\n".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "tool"
    )

    # A real model resolves the [S#] ids it was handed, whether they arrived in
    # a tool result (researcher) or in the dossier it is now analysing.  Read
    # both so the analyst and writer see the same registry the researcher did.
    citation_text = "\n".join(part for part in (tool_text, user_text) if part)
    seen: set[str] = set()
    sources: list[tuple[str, str, str]] = []
    for cid in CITATION_RE.findall(citation_text):
        if cid in seen:
            continue
        seen.add(cid)
        sources.append((cid, f"Source {cid}", "a registered offline source"))

    mock = MockLLM(model="mock/deterministic")
    content = mock.reply(system_text, user_text, sources=sources)
    return {"role": "assistant", "content": content}


class _Handler(BaseHTTPRequestHandler):
    state: _FakeState = None  # type: ignore[assignment]

    def log_message(self, *args: Any) -> None:  # silence the server
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}

        self.state.requests.append(payload)
        messages = payload.get("messages", [])
        tools = payload.get("tools")
        message = _openrouter_response(messages, tools)

        body = {
            "id": "gen-test",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": payload.get("model", "test-model"),
            "system_fingerprint": "fp-test",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@pytest.fixture
def fake_openrouter() -> Iterator[str]:
    """Start a fake OpenRouter server and yield its base URL."""
    state = _FakeState()
    handler = type("BoundHandler", (_Handler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}/api/v1"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def scrub_ambient_env(monkeypatch):
    """Stop the operator's shell or container environment leaking into tests.

    ``build_settings()`` reads ``os.environ`` even when ``env_file=None``, so an
    exported ``MARS_*`` override (a deployment host, a CI runner) would silently
    change the defaults under test.  Tests that need a variable set it
    themselves, after this fixture has run.
    """
    import os

    for name in list(os.environ):
        if name.startswith(_AMBIENT_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def isolated_topic_env(monkeypatch):
    """Keep the real ``.env`` from leaking an off-domain topic into tests.

    ``build_settings()`` reads ``os.environ``; the CLI tests exercise it, which
    would otherwise load the operator's ``.env`` once and leak into later
    tests.  Pin a valid Agentic AI topic for every test.
    """
    monkeypatch.setenv("MARS_TOPIC", "agentic AI")


@pytest.fixture(autouse=True)
def hermetic_llm_and_tools(monkeypatch):
    """Inject the test-only provider and corpus tool.

    Production has exactly one provider (OpenRouter) and no corpus.  Tests opt
    in through ``Settings.extra`` so the live-style contract test can still use
    the real client stack.
    """
    import mars.llm as llm_module
    import mars.pipeline as pipeline_module
    import mars.tools as tools_module
    from tests.support.fake_llm import MockLLM
    from tests.support.tools import TestCorpusTool

    real_build_llm = llm_module.build_llm
    real_build_toolset = tools_module.build_toolset

    import os

    def build_llm(settings, role="fast", *, usage=None, context=None):
        if settings.extra.get("fake_llm") or os.environ.get("MARS_TEST_FAKE_LLM") == "1":
            return MockLLM(usage=usage, label=role, context=context)
        return real_build_llm(settings, role, usage=usage, context=context)

    def build_toolset(context, *, track="general", include=None, settings=None):
        effective = settings or context.settings
        tools = real_build_toolset(context, track=track, include=include, settings=settings)
        if effective.extra.get("test_corpus") or os.environ.get("MARS_TEST_CORPUS") == "1":
            tools.append(TestCorpusTool(context=context))
        return tools

    monkeypatch.setattr(pipeline_module, "build_llm", build_llm)
    monkeypatch.setattr(tools_module, "build_toolset", build_toolset)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Hermetic settings: fake provider + test corpus, no outbound network."""
    return build_settings(
        topic="agentic AI",
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        cache_enabled=False,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        dead_letter_dir=tmp_path / "dead_letters",
        observability_dir=tmp_path / "traces",
        approval_dir=tmp_path / "approvals",
        archive_dir=tmp_path / "evidence",
        max_rounds=1,
        allow_network=False,
        env_file=None,
        extra={"fake_llm": True, "test_corpus": True},
    )


@pytest.fixture
def context(settings: Settings) -> Iterator[ResearchContext]:
    ctx = ResearchContext(settings=settings)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def network_context(settings: Settings) -> Iterator[ResearchContext]:
    """A context whose HTTP client may talk to a local stub server.

    Used by the tool tests; they point the tools at a loopback server, so no
    real outbound request is ever made.  Loopback destinations are only
    reachable when private networking is explicitly allowed.
    """
    network_settings = settings.with_overrides(
        allow_network=True, allow_private_network=True
    )
    ctx = ResearchContext(settings=network_settings)
    try:
        yield ctx
    finally:
        ctx.close()
