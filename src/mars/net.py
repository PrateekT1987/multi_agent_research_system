"""A polite, cached HTTP client used by every retrieval tool.

Features that matter for a research crawler:

* **Disk cache** - the same URL is fetched once per TTL, which keeps reruns
  cheap and makes the system reproducible.
* **Rate limiting** - a global requests-per-minute budget plus a per-host
  politeness delay.
* **Retries** - exponential backoff on timeouts, 429s and 5xx responses.
* **Egress safety** - only public ``http(s)`` destinations are fetched, every
  redirect is re-validated, response bodies are size-capped, and callers can
  restrict accepted content types.  This closes the SSRF path where a model or
  a malicious page could ask the crawler to read loopback, link-local or
  cloud-metadata endpoints.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import random
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

__all__ = ["HttpClient", "HttpError", "Response", "validate_url"]

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_BLOCKED_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "metadata.google.internal"})
_BLOCKED_SUFFIXES = (".local", ".internal", ".localhost", ".home.arpa")


class HttpError(RuntimeError):
    """Raised when a request cannot be completed after all retries."""


@dataclass(slots=True)
class Response:
    url: str
    status: int
    text: str
    headers: dict[str, str]
    from_cache: bool = False

    def json(self) -> Any:
        return json.loads(self.text)


def _blocked_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for loopback, private, link-local, reserved or multicast space."""
    return not address.is_global


def _resolve_host(host: str, port: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise HttpError(f"could not resolve host {host!r}: {exc}") from exc

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        raw = str(info[4][0]).split("%", 1)[0]
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise HttpError(f"could not resolve host {host!r} to an IP address")
    return addresses


def validate_url(url: str, *, allow_private_network: bool = False) -> None:
    """Refuse non-HTTP, credential-bearing or non-public destinations.

    DNS is resolved here so hostnames that point at private space are rejected
    before a connection is attempted.  A determined attacker can still rebind
    DNS between this check and the connect; the manual redirect validation in
    :meth:`HttpClient.request` is the complementary half of that defence.
    """
    if not isinstance(url, str) or not url.strip():
        raise HttpError("refusing to fetch an empty URL")
    parts = urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"}:
        raise HttpError(f"refusing non-HTTP URL: {url!r}")
    if not parts.hostname:
        raise HttpError(f"refusing URL without a host: {url!r}")
    if parts.username or parts.password:
        raise HttpError("refusing URL with embedded credentials")
    try:
        port = parts.port
    except ValueError as exc:
        raise HttpError(f"invalid port in URL: {url!r}") from exc

    host = parts.hostname.rstrip(".").lower()
    if allow_private_network:
        return
    if host in _BLOCKED_HOSTNAMES or host.endswith(_BLOCKED_SUFFIXES):
        raise HttpError(f"refusing private host {host!r}")

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        addresses = _resolve_host(host, port or (443 if parts.scheme.lower() == "https" else 80))
    else:
        addresses = [literal]

    for address in addresses:
        if _blocked_address(address):
            raise HttpError(
                f"refusing non-public address {address} for host {host!r} "
                "(set MARS_ALLOW_PRIVATE_NETWORK=1 only for local development)"
            )


class _RateLimiter:
    """Sliding-window limiter plus a per-host minimum spacing."""

    def __init__(self, requests_per_minute: int, min_seconds_between: float) -> None:
        self._limit = max(1, requests_per_minute)
        self._min_gap = max(0.0, min_seconds_between)
        self._times: list[float] = []
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()

    def acquire(self, host: str = "") -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._times = [t for t in self._times if now - t < 60.0]
                host_wait = 0.0
                if host:
                    last = self._last_request.get(host)
                    if last is not None:
                        host_wait = max(0.0, self._min_gap - (now - last))
                if len(self._times) < self._limit and host_wait <= 0:
                    self._times.append(now)
                    if host:
                        self._last_request[host] = now
                    return
                if len(self._times) >= self._limit:
                    wait = 60.0 - (now - self._times[0]) + 0.05
                else:
                    wait = host_wait + 0.05
            time.sleep(min(wait, 60.0))

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "requests_per_minute": self._limit,
                "min_seconds_between_requests": self._min_gap,
                "window_requests": len(self._times),
            }


_LIMITER_REGISTRY: dict[tuple[int, float], _RateLimiter] = {}
_LIMITER_LOCK = threading.Lock()


def get_shared_rate_limiter(
    requests_per_minute: int, min_seconds_between_requests: float
) -> _RateLimiter:
    """Return the process-wide limiter for a given policy.

    Concurrent runs in one process must share the outbound budget; otherwise
    an outage multiplies both retries and request rate by the run count.
    """
    key = (max(1, requests_per_minute), max(0.0, min_seconds_between_requests))
    with _LIMITER_LOCK:
        limiter = _LIMITER_REGISTRY.get(key)
        if limiter is None:
            limiter = _RateLimiter(*key)
            _LIMITER_REGISTRY[key] = limiter
        return limiter


class HttpClient:
    """Thread-safe HTTP client with an on-disk cache and egress policy."""

    def __init__(
        self,
        *,
        timeout: float = 25.0,
        user_agent: str = "MarsResearchBot/0.1",
        cache_dir: Path | None = None,
        cache_enabled: bool = True,
        cache_ttl_hours: int = 24,
        cache_max_bytes: int = 500_000_000,
        cache_max_entries: int = 50_000,
        requests_per_minute: int = 40,
        min_seconds_between_requests: float = 0.6,
        max_retries: int = 3,
        offline: bool = False,
        allow_network: bool = True,
        allow_private_network: bool = False,
        max_response_bytes: int = 10_000_000,
        max_redirects: int = 5,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl_hours * 3600
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_max_bytes = max(1, cache_max_bytes)
        self.cache_max_entries = max(1, cache_max_entries)
        self.max_retries = max(0, max_retries)
        self.allow_network = bool(allow_network) and not offline
        self.allow_private_network = bool(allow_private_network)
        self.max_response_bytes = max(1, max_response_bytes)
        self.max_redirects = max(0, max_redirects)
        self.offline = not self.allow_network
        self._limiter = get_shared_rate_limiter(requests_per_minute, min_seconds_between_requests)
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            headers={"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.9"},
        )
        self._cache_lock = threading.Lock()
        self.stats = {"requests": 0, "cache_hits": 0, "errors": 0, "bytes": 0}

    # ------------------------------------------------------------------ cache
    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_enabled or not self.cache_dir:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        path = self.cache_dir / "http" / digest[:2] / f"{digest}.json"
        return path

    def _cache_read(self, key: str) -> Response | None:
        path = self._cache_path(key)
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if self.cache_ttl and (time.time() - payload.get("fetched_at", 0)) > self.cache_ttl:
            return None
        return Response(
            url=payload.get("url", ""),
            status=payload.get("status", 200),
            text=payload.get("text", ""),
            headers=payload.get("headers", {}),
            from_cache=True,
        )

    def _cache_write(self, key: str, response: Response) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        try:
            from .storage import atomic_write_json

            atomic_write_json(
                path,
                {
                    "schema_version": 1,
                    "url": response.url,
                    "status": response.status,
                    "text": response.text,
                    "headers": response.headers,
                    "fetched_at": time.time(),
                },
            )
            self._prune_cache()
        except OSError:
            pass

    def _prune_cache(self) -> None:
        """Evict oldest cache entries until both size and count are bounded."""
        if self.cache_dir is None:
            return
        root = self.cache_dir / "http"
        if not root.is_dir():
            return
        with self._cache_lock:
            entries: list[tuple[float, int, Path]] = []
            total = 0
            for path in root.rglob("*.json"):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                entries.append((stat.st_mtime, stat.st_size, path))
                total += stat.st_size
            if len(entries) <= self.cache_max_entries and total <= self.cache_max_bytes:
                return
            entries.sort(key=lambda item: item[0])
            index = 0
            while entries and (
                len(entries) > self.cache_max_entries or total > self.cache_max_bytes
            ):
                if index >= len(entries):
                    break
                _, size, path = entries[index]
                try:
                    path.unlink()
                    total -= size
                    entries.pop(index)
                except OSError:
                    index += 1

    def cache_stats(self) -> dict[str, int]:
        """Current on-disk cache footprint."""
        if self.cache_dir is None:
            return {"entries": 0, "bytes": 0}
        root = self.cache_dir / "http"
        entries = 0
        total = 0
        if root.is_dir():
            for path in root.rglob("*.json"):
                try:
                    total += path.stat().st_size
                    entries += 1
                except OSError:
                    continue
        return {"entries": entries, "bytes": total}

    # ----------------------------------------------------------------- fetch
    def _send_with_retries(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None,
        data: dict[str, Any] | None,
        json_body: Any,
        headers: dict[str, str] | None,
        timeout: float | None,
    ) -> Response:
        host = urlsplit(url).hostname or ""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._limiter.acquire(host)
            try:
                with self._client.stream(
                    method,
                    url,
                    params=params,
                    data=data,
                    json=json_body,
                    headers=headers,
                    timeout=timeout or self.timeout,
                ) as response:
                    status = response.status_code
                    response_headers = dict(response.headers)
                    encoding = response.encoding or "utf-8"
                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > self.max_response_bytes:
                            raise HttpError(
                                f"response from {url} exceeded "
                                f"{self.max_response_bytes} bytes"
                            )
                        chunks.append(chunk)
                    content = b"".join(chunks)
                    final_url = str(response.url)

                self.stats["requests"] += 1
                self.stats["bytes"] += len(content)

                if status in (429, 500, 502, 503, 504):
                    last_error = HttpError(f"HTTP {status} for {url}")
                    if attempt < self.max_retries:
                        time.sleep(min(2**attempt + random.random(), 8.0))
                        continue
                    raise last_error

                return Response(
                    url=final_url,
                    status=status,
                    text=content.decode(encoding, errors="replace"),
                    headers=response_headers,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt + random.random(), 8.0))
                    continue
            except HttpError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                break

        self.stats["errors"] += 1
        raise HttpError(f"request failed for {url}: {last_error}")

    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
        timeout: float | None = None,
        allowed_content_types: set[str] | None = None,
    ) -> Response:
        """Perform an HTTP request with egress checks, redirects and caching."""
        if not self.allow_network:
            raise HttpError(
                f"network access is disabled; refusing to make a request to {url}"
            )

        validate_url(url, allow_private_network=self.allow_private_network)
        cache_key = json.dumps(
            [method.upper(), url, params or {}, data or {}, json_body],
            sort_keys=True,
            default=str,
        )
        if use_cache:
            cached = self._cache_read(cache_key)
            if cached is not None:
                self.stats["cache_hits"] += 1
                return cached

        current_url = url
        current_method = method.upper()
        current_params = params
        current_data = data
        current_json = json_body
        redirects = 0

        while True:
            validate_url(current_url, allow_private_network=self.allow_private_network)
            response = self._send_with_retries(
                current_method,
                current_url,
                params=current_params,
                data=current_data,
                json_body=current_json,
                headers=headers,
                timeout=timeout,
            )

            location = response.headers.get("location")
            if response.status in _REDIRECT_STATUSES and location:
                if redirects >= self.max_redirects:
                    raise HttpError(f"too many redirects for {url}")
                next_url = urljoin(current_url, location)
                validate_url(next_url, allow_private_network=self.allow_private_network)
                if response.status == 303 or (
                    response.status in (301, 302) and current_method not in {"GET", "HEAD"}
                ):
                    current_method = "GET"
                    current_data = None
                    current_json = None
                current_params = None
                current_url = next_url
                redirects += 1
                continue

            if allowed_content_types:
                content_type = (
                    response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                )
                if content_type and content_type not in allowed_content_types:
                    raise HttpError(
                        f"refusing content type {content_type!r} from {current_url}"
                    )

            if use_cache and response.status == 200:
                self._cache_write(cache_key, response)
            return response

    def get(self, url: str, **kwargs: Any) -> Response:
        return self.request(url, method="GET", **kwargs)

    def post(self, url: str, **kwargs: Any) -> Response:
        return self.request(url, method="POST", **kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
