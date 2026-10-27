"""Tests for the HTTP client: caching, retries and rate limiting."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mars.net import HttpClient, HttpError


class _Counter(BaseHTTPRequestHandler):
    hits = 0
    lock = threading.Lock()

    def log_message(self, *args: object) -> None:
        return

    def do_GET(self) -> None:
        with _Counter.lock:
            _Counter.hits += 1
            hit = _Counter.hits
        if self.path == "/flaky" and hit < 3:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        payload = json.dumps({"hit": hit}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def server():
    _Counter.hits = 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Counter)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cache_avoids_a_second_request(server, tmp_path) -> None:
    client = HttpClient(
        cache_dir=tmp_path,
        cache_enabled=True,
        min_seconds_between_requests=0,
        allow_private_network=True,
    )
    first = client.get(f"{server}/thing")
    second = client.get(f"{server}/thing")

    assert first.text == second.text
    assert second.from_cache is True
    assert client.stats["cache_hits"] == 1


def test_retry_on_server_error(server, tmp_path) -> None:
    client = HttpClient(
        cache_dir=tmp_path,
        cache_enabled=False,
        max_retries=3,
        min_seconds_between_requests=0,
        allow_private_network=True,
    )
    response = client.get(f"{server}/flaky")
    assert response.status == 200
    assert client.stats["requests"] == 3


def test_failure_raises_http_error(server, tmp_path) -> None:
    client = HttpClient(
        cache_dir=tmp_path,
        cache_enabled=False,
        max_retries=1,
        min_seconds_between_requests=0,
        allow_private_network=True,
    )
    with pytest.raises(HttpError):
        client.get("http://127.0.0.1:1/nope")
