"""Run metrics: token usage, timings, tool activity and cost estimates."""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "UsageTracker",
    "PRICING",
    "SpendBudgetExceeded",
    "current_agent",
    "current_track",
    "estimate_cost",
    "estimate_run_cost",
    "pricing_known",
    "usage_scope",
    "Stopwatch",
]


class SpendBudgetExceeded(RuntimeError):
    """A run has exhausted a token or currency ceiling."""


# The active agent/track is carried in a context variable so parallel tracks
# cannot attribute each other's calls.  The pipeline sets a scope around every
# agent invocation; the LLM reads it at record time.
_AGENT: contextvars.ContextVar[str] = contextvars.ContextVar("mars_agent", default="unknown")
_TRACK: contextvars.ContextVar[str] = contextvars.ContextVar("mars_track", default="run")


def current_agent() -> str:
    return _AGENT.get()


def current_track() -> str:
    return _TRACK.get()


@contextmanager
def usage_scope(*, agent: str | None = None, track: str | None = None) -> Iterator[None]:
    """Attribute LLM calls made inside the block to an agent and/or track."""
    tokens: list[tuple[contextvars.ContextVar[str], contextvars.Token[str]]] = []
    if agent is not None:
        tokens.append((_AGENT, _AGENT.set(agent)))
    if track is not None:
        tokens.append((_TRACK, _TRACK.set(track)))
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


# USD per token. OpenRouter reports the same unit, so the numbers map directly.
PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4.5": (3e-6, 15e-6),
    "anthropic/claude-opus-4.1": (15e-6, 75e-6),
    "anthropic/claude-haiku-4.5": (1e-6, 5e-6),
    "openai/gpt-4.1": (2e-6, 8e-6),
    "openai/gpt-4.1-mini": (4e-7, 1.6e-6),
    "openai/gpt-5-mini": (2.5e-7, 2e-6),
    "openai/gpt-5": (1.25e-6, 10e-6),
    "google/gemini-2.5-flash": (3e-7, 2.5e-6),
    "google/gemini-2.5-pro": (1.25e-6, 1e-5),
    "deepseek/deepseek-chat": (2.6e-7, 1.03e-6),
    # The configured production model is a free OpenRouter tier.  Listing it
    # explicitly keeps it distinguishable from "we have no idea what this is".
    "nvidia/nemotron-3-ultra-550b-a55b:free": (0.0, 0.0),
}

# Rough per-depth token envelopes used by the UI estimate and documentation.
_DEPTH_TOKENS = {"quick": 25_000, "standard": 90_000, "deep": 220_000}


def pricing_known(model: str) -> bool:
    """Whether a model has an explicit price entry."""
    return model in PRICING


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Best-effort USD estimate for a model call."""
    prompt_price, completion_price = PRICING.get(model, (0.0, 0.0))
    return prompt_tokens * prompt_price + completion_tokens * completion_price


def estimate_run_cost(model: str, depth: str) -> tuple[float | None, int]:
    """Estimate USD and tokens for a depth preset, or ``None`` if unpriced."""
    tokens = _DEPTH_TOKENS.get(depth, _DEPTH_TOKENS["standard"])
    if not pricing_known(model):
        return None, tokens
    # Rough 70/30 prompt/completion split for research-style workloads.
    return estimate_cost(model, int(tokens * 0.7), int(tokens * 0.3)), tokens


@dataclass
class UsageTracker:
    """Thread-safe accumulator of LLM usage and spend ceilings."""

    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_tokens_per_track: int | None = None
    max_cost_usd_per_track: float | None = None
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_role: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_agent: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_track: dict[str, dict[str, Any]] = field(default_factory=dict)
    unpriced_models: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @staticmethod
    def _bump(
        mapping: dict[str, dict[str, Any]],
        key: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
    ) -> None:
        bucket = mapping.setdefault(
            key, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += prompt_tokens
        bucket["completion_tokens"] += completion_tokens
        bucket["cost"] += cost

    def record(
        self,
        *,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        elapsed: float = 0.0,
        cached: bool = False,
        role: str = "unknown",
        agent: str | None = None,
        track: str | None = None,
    ) -> None:
        resolved_agent = agent or current_agent()
        resolved_track = track or current_track()
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        with self._lock:
            if not pricing_known(model):
                self.unpriced_models.add(model)
            self.calls += 1
            if cached:
                self.cached_calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.elapsed += elapsed
            self._bump(
                self.by_model,
                model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
            )
            self._bump(
                self.by_role,
                role,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
            )
            self._bump(
                self.by_agent,
                resolved_agent,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
            )
            self._bump(
                self.by_track,
                resolved_track,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
            )

    def check_budget(self) -> None:
        """Raise before the next call if a run or track ceiling is spent."""
        with self._lock:
            if self.max_tokens is not None and self.total_tokens >= self.max_tokens:
                raise SpendBudgetExceeded(
                    f"token budget exhausted ({self.total_tokens}/{self.max_tokens})"
                )
            if self.max_cost_usd is not None and self.cost >= self.max_cost_usd:
                raise SpendBudgetExceeded(
                    f"cost budget exhausted (${self.cost:.4f}/${self.max_cost_usd:.4f})"
                )

            track = current_track()
            bucket = self.by_track.get(track, {})
            track_tokens = int(bucket.get("prompt_tokens", 0)) + int(
                bucket.get("completion_tokens", 0)
            )
            track_cost = float(bucket.get("cost", 0.0))
            if (
                self.max_tokens_per_track is not None
                and track_tokens >= self.max_tokens_per_track
            ):
                raise SpendBudgetExceeded(
                    f"track {track!r} token budget exhausted "
                    f"({track_tokens}/{self.max_tokens_per_track})"
                )
            if (
                self.max_cost_usd_per_track is not None
                and track_cost >= self.max_cost_usd_per_track
            ):
                raise SpendBudgetExceeded(
                    f"track {track!r} cost budget exhausted "
                    f"(${track_cost:.4f}/${self.max_cost_usd_per_track:.4f})"
                )

    @staticmethod
    def _merge_mapping(
        target: dict[str, dict[str, Any]],
        source: dict[str, dict[str, Any]],
    ) -> None:
        for key, bucket in source.items():
            destination = target.setdefault(
                key, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
            )
            for name, value in bucket.items():
                destination[name] = destination.get(name, 0) + value

    def merge(self, other: UsageTracker) -> None:
        with self._lock, other._lock:
            self.calls += other.calls
            self.cached_calls += other.cached_calls
            self.prompt_tokens += other.prompt_tokens
            self.completion_tokens += other.completion_tokens
            self.elapsed += other.elapsed
            self.unpriced_models.update(other.unpriced_models)
            self._merge_mapping(self.by_model, other.by_model)
            self._merge_mapping(self.by_role, other.by_role)
            self._merge_mapping(self.by_agent, other.by_agent)
            self._merge_mapping(self.by_track, other.by_track)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost(self) -> float:
        return sum(bucket.get("cost", 0.0) for bucket in self.by_model.values())

    @staticmethod
    def _bucket_snapshot(mapping: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "calls": bucket["calls"],
                "prompt_tokens": bucket["prompt_tokens"],
                "completion_tokens": bucket["completion_tokens"],
                "estimated_cost_usd": round(bucket.get("cost", 0.0), 6),
            }
            for key, bucket in mapping.items()
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "cached_calls": self.cached_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.cost, 6),
            "cost_known": not self.unpriced_models,
            "unpriced_models": sorted(self.unpriced_models),
            "budget": {
                "max_tokens": self.max_tokens,
                "max_cost_usd": self.max_cost_usd,
                "max_tokens_per_track": self.max_tokens_per_track,
                "max_cost_usd_per_track": self.max_cost_usd_per_track,
                "tokens_remaining": (
                    None if self.max_tokens is None else max(0, self.max_tokens - self.total_tokens)
                ),
                "cost_remaining_usd": (
                    None
                    if self.max_cost_usd is None
                    else round(max(0.0, self.max_cost_usd - self.cost), 6)
                ),
            },
            "llm_seconds": round(self.elapsed, 2),
            "by_model": self._bucket_snapshot(self.by_model),
            "by_role": self._bucket_snapshot(self.by_role),
            "by_agent": self._bucket_snapshot(self.by_agent),
            "by_track": self._bucket_snapshot(self.by_track),
        }


class Stopwatch:
    """Context manager that reports wall-clock duration."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.started = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> Stopwatch:
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.started
