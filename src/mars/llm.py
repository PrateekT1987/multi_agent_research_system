"""LLM providers for Mars.

Mars talks to models through a small abstraction so the rest of the system
never depends on a vendor SDK.  There is exactly one production provider:
:class:`OpenRouterLLM`.  It is implemented as a ``crewai.BaseLLM`` so CrewAI
can drive it directly, and it is backed by ``langchain-openrouter`` (the
OpenAI-compatible endpoint OpenRouter exposes).

Every prompt is passed through the data-governance redactor before it leaves
the process, so secrets and (optionally) personal data are not disclosed to
the model provider.  Test-only deterministic providers live under
``tests/support`` and are never imported by production code.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from crewai.llm import BaseLLM
from pydantic import ConfigDict, Field, PrivateAttr

from .config import Settings
from .metrics import UsageTracker, current_agent, current_track
from .redaction import redact_messages
from .reliability import (
    CallBudget,
    CallBudgetExceeded,
    CircuitOpenError,
    ProviderResponseError,
    ProviderUnavailableError,
    SharedCircuitBreaker,
)

__all__ = ["OpenRouterLLM", "build_llm", "ToolCall", "FunctionCall"]

#: Provider codes that describe a saturated or briefly broken upstream.  These
#: are safe to retry with backoff and should count against the shared breaker.
_TRANSIENT_ENVELOPE_CODES = frozenset(
    {
        "provider_overloaded",
        "overloaded",
        "server_error",
        "internal_server_error",
        "timeout",
        "unavailable",
        "upstream_error",
    }
)
#: Provider codes that describe a rate limit (which may be transient or a
#: spent daily quota, depending on the message).
_RATE_LIMIT_ENVELOPE_CODES = frozenset(
    {"rate_limit_exceeded", "rate_limited", "rate_limit", "too_many_requests"}
)
#: Provider codes that mean the run cannot continue with this account/model.
_QUOTA_ENVELOPE_CODES = frozenset({"insufficient_quota", "quota_exceeded"})



# ---------------------------------------------------------------------------
# Tool-call containers understood by CrewAI's native tool-calling executor
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FunctionCall:
    name: str
    arguments: str


@dataclass(slots=True)
class ToolCall:
    id: str
    function: FunctionCall
    type: str = "function"


def _usage_tokens(message: Any) -> tuple[int, int]:
    """Return ``(prompt, completion)`` tokens from a LangChain message.

    LangChain exposes usage as ``usage_metadata`` on the message; some clients
    only populate ``response_metadata``.  Returning zeros when neither exists is
    deliberate: the call is still counted, and the tracker treats unknown usage
    as zero rather than inventing a number.
    """
    usage = getattr(message, "usage_metadata", None) or {}
    if usage:
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
        )
    metadata = getattr(message, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
    return (
        int(token_usage.get("prompt_tokens", 0) or 0),
        int(token_usage.get("completion_tokens", 0) or 0),
    )


class _LLMBase(BaseLLM):
    """Shared behaviour: usage tracking, disk cache, prompt rendering."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    usage: UsageTracker = Field(default_factory=UsageTracker)
    cache_dir: Path | None = Field(default=None, exclude=True)
    cache_enabled: bool = Field(default=False, exclude=True)
    cache_ttl_hours: float = Field(default=168.0, exclude=True)
    cache_max_entries: int = Field(default=10_000, exclude=True)
    label: str = Field(default="llm", exclude=True)
    redact_pii: bool = Field(default=True, exclude=True)

    _counter: int = PrivateAttr(default=0)

    # -------------------------------------------------------------- helpers
    def supports_function_calling(self) -> bool:  # pragma: no cover - overridden
        return False

    def supports_stop_words(self) -> bool:
        return False

    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_enabled or not self.cache_dir:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return Path(self.cache_dir) / "llm" / digest[:2] / f"{digest}.json"

    def _cache_read(self, key: str) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if self.cache_ttl_hours and self.cache_ttl_hours > 0:
            age = time.time() - float(payload.get("cached_at", 0) or 0)
            if age > self.cache_ttl_hours * 3600:
                return None
        return payload

    def _cache_write(self, key: str, payload: dict[str, Any]) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        try:
            from .storage import atomic_write_json

            atomic_write_json(path, {**payload, "schema_version": 1, "cached_at": time.time()})
            self._prune_cache()
        except OSError:
            pass

    def _prune_cache(self) -> None:
        """Bound the LLM cache by entry count, evicting the oldest first."""
        if not self.cache_dir:
            return
        root = Path(self.cache_dir) / "llm"
        if not root.is_dir():
            return
        entries: list[tuple[float, Path]] = []
        for path in root.rglob("*.json"):
            try:
                entries.append((path.stat().st_mtime, path))
            except OSError:
                continue
        if len(entries) <= self.cache_max_entries:
            return
        entries.sort(key=lambda item: item[0])
        for _, path in entries[: len(entries) - self.cache_max_entries]:
            try:
                path.unlink()
            except OSError:
                continue

    def _cache_key(self, *parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _render(self, messages: Any) -> list[dict[str, Any]]:
        """Normalise CrewAI messages into OpenAI-style dicts."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        rendered: list[dict[str, Any]] = []
        for message in messages or []:
            if isinstance(message, dict):
                item = dict(message)
            else:
                item = {
                    "role": getattr(message, "role", "user"),
                    "content": getattr(message, "content", ""),
                }
            item.setdefault("role", "user")
            item.setdefault("content", "")
            rendered.append(item)
        return redact_messages(rendered, redact_pii=self.redact_pii)


class OpenRouterLLM(_LLMBase):
    """OpenRouter-hosted chat models, driven through ``langchain-openrouter``."""

    llm_type: str = "openrouter"
    model: str
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float | None = 0.25
    max_tokens: int | None = 8192
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)
    max_retries: int = 3
    """Retry budget handed to the provider.

    ``langchain-openrouter`` gives every retry unit a ~150 second backoff
    window, so the value is capped when the client is built: an unbounded
    retry budget turns one flaky call into a multi-minute stall.
    """
    request_timeout: float = 120.0
    extra_headers: dict[str, str] = Field(default_factory=dict)
    breaker: SharedCircuitBreaker | None = Field(default=None, exclude=True)
    call_budget: CallBudget | None = Field(default=None, exclude=True)
    retry_base_seconds: float = Field(default=1.0, exclude=True)
    retry_max_seconds: float = Field(default=20.0, exclude=True)
    # How long one call waits for a shared breaker to leave its open state.
    # ``0`` disables waiting.  A positive value is what turns a provider blip
    # from a run-wide failure into a pause: every parallel agent waits on the
    # same breaker, one probe goes through, and the fleet resumes when it
    # succeeds.
    circuit_wait_seconds: float = Field(default=40.0, exclude=True)
    observer: Any = Field(default=None, exclude=True)
    watchdog: Any = Field(default=None, exclude=True)

    _client: Any = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def model_post_init(self, __context: Any) -> None:
        self._client = None

    # ------------------------------------------------------------------ setup
    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            from langchain_openrouter import ChatOpenRouter

            kwargs: dict[str, Any] = {
                "model": self.model,
                "temperature": self.temperature,
                "base_url": self.base_url,
                # Mars owns retries so every attempt is budgeted, jittered and
                # visible to the run-wide circuit breaker. Never layer SDK
                # retries underneath it.
                "max_retries": 0,
                # langchain-openrouter takes the timeout in *milliseconds* and
                # maps it onto the SDK's ``timeout_ms``.  Passing seconds here
                # silently turns a 120s budget into a 120ms one, which looks
                # like the provider is hanging rather than timing out.
                "timeout": max(1, int(self.request_timeout * 1000)),
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.frequency_penalty is not None:
                kwargs["frequency_penalty"] = self.frequency_penalty
            if self.presence_penalty is not None:
                kwargs["presence_penalty"] = self.presence_penalty
            if self.seed is not None:
                kwargs["seed"] = self.seed
            if self.stop:
                kwargs["stop"] = self.stop
            if self.extra_headers:
                kwargs["default_headers"] = self.extra_headers
            self._client = ChatOpenRouter(**kwargs)
            return self._client

    def supports_function_calling(self) -> bool:
        return True

    @staticmethod
    def _provider_envelope(exc: Exception) -> dict[str, str] | None:
        """Return ``{"code", "message"}`` when the response was an error body.

        OpenRouter occasionally answers ``200 OK`` with
        ``{"id": "gen-…", "error": {"code": "provider_overloaded", …}}``.  The
        Speakeasy SDK tries to unmarshal that into a chat completion, fails
        Pydantic validation, and raises ``ResponseValidationError`` whose
        ``body`` still holds the original JSON.  Reading that envelope is what
        lets Mars retry a transient upstream failure instead of treating a
        schema mismatch as a permanent client error.
        """
        candidates: list[Any] = []
        body = getattr(exc, "body", None)
        if isinstance(body, (str, bytes)):
            candidates.append(body)
        response = getattr(exc, "response", None) or getattr(exc, "raw_response", None)
        text = getattr(response, "text", None)
        if isinstance(text, (str, bytes)):
            candidates.append(text)
        cause = exc.__cause__
        errors = getattr(cause, "errors", None)
        if callable(errors):
            try:
                for item in errors():
                    if isinstance(item, dict) and isinstance(item.get("input"), dict):
                        candidates.append(item["input"])
            except Exception:  # pragma: no cover - defensive
                pass

        for candidate in candidates:
            payload: Any = candidate
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", "replace")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except (TypeError, ValueError):
                    continue
            if not isinstance(payload, dict):
                continue
            error = payload.get("error")
            if not isinstance(error, dict):
                continue
            code = str(error.get("code") or error.get("type") or "").strip().lower()
            if not code:
                continue
            return {"code": code, "message": str(error.get("message") or "").strip()}
        return None

    @classmethod
    def _envelope_code(cls, exc: Exception) -> str | None:
        envelope = cls._provider_envelope(exc)
        return envelope["code"] if envelope else None

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        status = getattr(exc, "status_code", None)
        if status is None:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
        return status if isinstance(status, int) else None

    @classmethod
    def _is_rate_limit(cls, exc: Exception) -> bool:
        if cls._envelope_code(exc) in _RATE_LIMIT_ENVELOPE_CODES:
            return True
        if cls._status_code(exc) == 429:
            return True
        text = str(exc).lower()
        return "429" in text or "rate limit" in text or "too many requests" in text

    @classmethod
    def _quota_exhausted(cls, exc: Exception) -> bool:
        """True when a 429 is a hard quota, not a short-window limit.

        Retrying a spent daily free-tier allowance only delays the failure and
        burns the run's call budget, so these errors are fatal and carry an
        actionable message instead.
        """
        if cls._envelope_code(exc) in _QUOTA_ENVELOPE_CODES:
            return True
        if not cls._is_rate_limit(exc):
            return False
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "per-day",
                "per day",
                "daily",
                "quota",
                "credit",
                "insufficient",
                "free-models-per-day",
            )
        )

    @classmethod
    def _retryable(cls, exc: Exception) -> bool:
        """Only retry transient provider/network failures."""
        code = cls._envelope_code(exc)
        if code is not None:
            if code in _QUOTA_ENVELOPE_CODES:
                return False
            if code in _TRANSIENT_ENVELOPE_CODES:
                return True
            if code in _RATE_LIMIT_ENVELOPE_CODES:
                return not cls._quota_exhausted(exc)
            # An envelope with an unknown code is a protocol-level error, not a
            # blip: retrying it would only multiply the failure.
            return False
        if cls._quota_exhausted(exc):
            return False
        status = cls._status_code(exc)
        if status is not None:
            return status in {408, 409, 429} or status >= 500
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "timeout",
                "timed out",
                "connection",
                "temporar",
                "rate limit",
                "429",
                "502",
                "503",
                "504",
            )
        )

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        """Read a provider's ``Retry-After`` hint (seconds or HTTP date)."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
        if not headers:
            return None
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:  # pragma: no cover - exotic header containers
            return None
        if value is None:
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            try:
                when = parsedate_to_datetime(str(value))
                if when.tzinfo is None:
                    when = when.replace(tzinfo=UTC)
                seconds = (when - datetime.now(UTC)).total_seconds()
            except Exception:
                return None
        # Never let a hostile or broken header stall the run.
        return max(0.0, min(seconds, 120.0))

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        """Equal-jitter exponential backoff, preferring the provider's hint.

        Full jitter can pick a delay near zero, which re-hammers an endpoint
        that just rate-limited us.  Equal jitter keeps a floor while still
        spreading the parallel tracks apart.
        """
        hinted = self._retry_after_seconds(exc)
        if hinted is not None:
            return hinted
        ceiling = min(self.retry_max_seconds, self.retry_base_seconds * (2**attempt))
        if ceiling <= 0:
            return 0.0
        return ceiling / 2 + random.uniform(0.0, ceiling / 2)

    def _invoke_with_resilience(self, operation: Any) -> Any:
        """Run one provider operation with bounded, fleet-coordinated retries."""
        attempts = max(0, min(self.max_retries, 5)) + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            attempt_started = time.perf_counter()
            if self.watchdog is not None:
                self.watchdog.check_run()
            self.usage.check_budget()
            probe = False
            if self.breaker is not None:
                wait = self.circuit_wait_seconds
                if self.watchdog is not None:
                    # Never sleep past the run's own deadline.
                    wait = min(wait, self.watchdog.remaining_run_seconds())
                try:
                    probe = self.breaker.before_call(timeout=wait)
                except CircuitOpenError:
                    if self.observer is not None:
                        self.observer.emit(
                            "llm.attempt",
                            status="blocked",
                            role=self.label,
                            attempt=attempt + 1,
                            circuit=self.breaker.snapshot(),
                        )
                    raise
            if self.call_budget is not None:
                self.call_budget.reserve()
            try:
                result = operation()
            except (CircuitOpenError, CallBudgetExceeded):
                if probe and self.breaker is not None:
                    self.breaker.release_probe()
                if self.observer is not None:
                    self.observer.emit("llm.attempt", status="blocked", role=self.label, attempt=attempt + 1)
                raise
            except Exception as exc:
                last_error = exc
                envelope = self._provider_envelope(exc)
                retryable = self._retryable(exc)
                final_attempt = attempt == attempts - 1
                # Count one failure per *call*, not per attempt: a single call
                # retrying three times must not open the fleet-wide breaker by
                # itself.  Hard quota errors are not availability failures.
                if retryable and final_attempt and self.breaker is not None:
                    self.breaker.record_failure()
                elif probe and self.breaker is not None:
                    self.breaker.release_probe()
                if self.observer is not None:
                    self.observer.emit(
                        "llm.attempt",
                        status="error",
                        role=self.label,
                        attempt=attempt + 1,
                        elapsed_ms=round((time.perf_counter() - attempt_started) * 1000, 2),
                        error_type=type(exc).__name__,
                        error_code=(envelope or {}).get("code"),
                        retryable=retryable,
                        rate_limited=self._is_rate_limit(exc),
                        circuit=self.breaker.snapshot() if self.breaker else {},
                    )
                if not retryable or final_attempt:
                    if self._quota_exhausted(exc):
                        raise self._quota_error(exc) from exc
                    if envelope is not None:
                        raise ProviderResponseError(
                            envelope["code"],
                            envelope["message"],
                            provider=self.llm_type,
                        ) from exc
                    raise
                # Hand the probe back before backing off so the next attempt
                # can re-enter the breaker instead of waiting on itself.
                if probe and self.breaker is not None:
                    self.breaker.release_probe()
                time.sleep(self._retry_delay(exc, attempt))
                continue
            if self.breaker is not None:
                self.breaker.record_success()
            if self.observer is not None:
                self.observer.emit(
                    "llm.attempt",
                    role=self.label,
                    attempt=attempt + 1,
                    elapsed_ms=round((time.perf_counter() - attempt_started) * 1000, 2),
                    circuit=self.breaker.snapshot() if self.breaker else {},
                )
            return result
        raise last_error or RuntimeError("LLM invocation failed")

    def _quota_error(self, exc: Exception) -> ProviderUnavailableError:
        """Turn a spent free-tier quota into an actionable, non-retryable error."""
        return ProviderUnavailableError(
            f"OpenRouter rate limit for {self.model!r} is a hard quota, not a "
            f"transient limit ({str(exc)[:300]}). Wait for the quota window to "
            "reset, or configure a paid route/another model with "
            "MARS_MODEL_STRONG, MARS_MODEL_FAST and MARS_MODEL_PLANNER."
        )

    # ------------------------------------------------------------------ call
    def call(
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: type[Any] | None = None,
    ) -> str | Any:
        """Invoke the model and return text, a tool-call list, or a model."""
        rendered = self._render(messages)
        self.usage.check_budget()
        cache_key = self._cache_key(
            "openrouter",
            self.model,
            self.temperature,
            self.max_tokens,
            response_model.__name__ if response_model else None,
            rendered,
            tools,
        )
        cached = self._cache_read(cache_key)
        if cached is not None:
            self.usage.record(
                model=self.model,
                prompt_tokens=int(cached.get("prompt_tokens", 0)),
                completion_tokens=int(cached.get("completion_tokens", 0)),
                cached=True,
                role=self.label,
                agent=current_agent(),
                track=current_track(),
            )
            if cached.get("tool_calls"):
                return [
                    ToolCall(
                        id=item["id"],
                        function=FunctionCall(
                            name=item["name"], arguments=item.get("arguments", "{}")
                        ),
                    )
                    for item in cached["tool_calls"]
                ]
            if cached.get("structured") is not None:
                return response_model.model_validate(cached["structured"]) if response_model else None
            return cached.get("content", "")

        client = self._ensure_client()
        started = time.monotonic()

        if response_model is not None:
            result, prompt_tokens, completion_tokens = self._call_structured(
                client, rendered, response_model
            )
            self.usage.record(
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                elapsed=time.monotonic() - started,
                role=self.label,
                agent=current_agent(),
                track=current_track(),
            )
            if self.observer is not None:
                self.observer.emit(
                    "llm.usage",
                    role=self.label,
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    estimated_cost_usd=round(self.usage.cost, 6),
                )
            self._cache_write(
                cache_key,
                {
                    "content": "",
                    "structured": result.model_dump(mode="json"),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            )
            return result

        bound = client.bind_tools(tools) if tools else client
        response = self._invoke_with_resilience(lambda: bound.invoke(rendered))

        usage_meta = getattr(response, "usage_metadata", None) or {}
        prompt_tokens = int(usage_meta.get("input_tokens", 0) or 0)
        completion_tokens = int(usage_meta.get("output_tokens", 0) or 0)
        self.usage.record(
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed=time.monotonic() - started,
            role=self.label,
            agent=current_agent(),
            track=current_track(),
        )
        if self.observer is not None:
            self.observer.emit(
                "llm.usage",
                role=self.label,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=round(self.usage.cost, 6),
            )

        raw_calls = getattr(response, "tool_calls", None) or []
        if raw_calls:
            calls: list[dict[str, Any]] = []
            for index, call in enumerate(raw_calls):
                arguments = call.get("args", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                calls.append(
                    {
                        "id": call.get("id") or f"call_{index}",
                        "name": call.get("name", ""),
                        "arguments": arguments,
                    }
                )
            self._cache_write(
                cache_key,
                {
                    "content": "",
                    "tool_calls": calls,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            )
            return [
                ToolCall(
                    id=item["id"],
                    function=FunctionCall(name=item["name"], arguments=item["arguments"]),
                )
                for item in calls
            ]

        content = response.content if isinstance(response.content, str) else str(response.content)
        self._cache_write(
            cache_key,
            {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
        return content

    def _call_structured(
        self, client: Any, rendered: list[dict[str, Any]], response_model: type[Any]
    ) -> tuple[Any, int, int]:
        """Force a JSON object matching ``response_model``.

        Returns the parsed model and the provider-reported token counts so the
        run's token/cost ceilings also cover structured calls (director,
        analyst and auditor).  ``include_raw`` retains the underlying message,
        which is where LangChain exposes usage metadata.
        """
        try:
            structured = client.with_structured_output(response_model, include_raw=True)
            outcome = self._invoke_with_resilience(lambda: structured.invoke(rendered))
            if isinstance(outcome, dict) and "parsed" in outcome:
                prompt_tokens, completion_tokens = _usage_tokens(outcome.get("raw"))
                parsed = outcome.get("parsed")
                if parsed is not None:
                    return parsed, prompt_tokens, completion_tokens
            elif outcome is not None:
                return outcome, 0, 0
        except (CircuitOpenError, CallBudgetExceeded):
            raise
        except ProviderResponseError:
            # The provider is overloaded or the request is invalid; a second
            # JSON-only call would fail the same way and double the cost.
            raise
        except Exception as exc:
            if self._quota_exhausted(exc):
                raise self._quota_error(exc) from exc
            # Structured output is not universally supported. Its failed,
            # bounded invocation has already been accounted for; use one
            # explicit JSON-only fallback rather than an unbounded repair loop.
            pass

        from .schemas import parse_json_block

        schema_hint = json.dumps(response_model.model_json_schema(), indent=2)[:4000]
        prompt = [
            *rendered,
            {
                "role": "user",
                "content": (
                    "Respond with a single JSON object that validates against this "
                    f"schema. Output JSON only.\n\n{schema_hint}"
                ),
            },
        ]
        raw = self._invoke_with_resilience(lambda: client.invoke(prompt))
        prompt_tokens, completion_tokens = _usage_tokens(raw)
        content = raw.content if isinstance(raw.content, str) else str(raw.content)
        return (
            response_model.model_validate(parse_json_block(content)),
            prompt_tokens,
            completion_tokens,
        )



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_ROLE_MODELS = {
    "planner": "model_planner",
    "strong": "model_strong",
    "fast": "model_fast",
}


def build_llm(
    settings: Settings,
    role: str = "fast",
    *,
    usage: UsageTracker | None = None,
    context: Any = None,
) -> BaseLLM:
    """Create the LLM used for a given agent role."""
    model_attr = _ROLE_MODELS.get(role, "model_fast")
    model = getattr(settings, model_attr, settings.model_fast)

    return OpenRouterLLM(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        max_retries=settings.max_llm_retries,
        request_timeout=settings.request_timeout,
        breaker=getattr(context, "llm_breaker", None),
        call_budget=getattr(context, "llm_budget", None),
        retry_base_seconds=settings.retry_base_seconds,
        retry_max_seconds=settings.retry_max_seconds,
        circuit_wait_seconds=settings.circuit_wait_seconds,
        observer=getattr(context, "observer", None),
        watchdog=getattr(context, "watchdog", None),
        usage=usage or UsageTracker(),
        cache_dir=settings.cache_dir,
        cache_enabled=settings.cache_enabled,
        cache_ttl_hours=settings.llm_cache_ttl_hours,
        cache_max_entries=settings.llm_cache_max_entries,
        redact_pii=settings.redact_pii,
        label=role,
        extra_headers={
            "HTTP-Referer": "https://github.com/mars-research/mars",
            "X-Title": "Mars Research System",
        },
    )
