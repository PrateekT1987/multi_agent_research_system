"""Pinned model registry and lifecycle policy.

Mars sends every model call to an exact OpenRouter model id.  Pinning is not
enough on its own: providers deprecate and retire models, and a run that
silently changes model is not reproducible.  This registry records the status
of every model Mars is allowed to use, the replacement to migrate to, and the
date the provider announced the change.

The registry is intentionally small.  A model that is not listed still works
(OpenRouter routes many ids), but it is reported as ``unpinned`` so an operator
can add it before a release.  Retired models fail validation; deprecated models
produce a startup warning with their replacement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

__all__ = [
    "DEFAULT_MODEL",
    "MODEL_REGISTRY",
    "ModelPin",
    "ModelRegistryError",
    "model_manifest",
    "model_pin",
    "resolve_model_chain",
    "validate_model",
]

ModelStatus = Literal["active", "deprecated", "retired"]

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


class ModelRegistryError(ValueError):
    """A configured model is retired or disallowed by policy."""


@dataclass(frozen=True, slots=True)
class ModelPin:
    """One exact model id and its lifecycle metadata."""

    model_id: str
    provider: str = "openrouter"
    status: ModelStatus = "active"
    released: str = ""
    deprecation_date: str | None = None
    replacement: str | None = None
    context_window: int | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# Keep this list curated.  Add a pin before changing a production default; the
# release checklist in README.md depends on it.
MODEL_REGISTRY: dict[str, ModelPin] = {
    DEFAULT_MODEL: ModelPin(
        model_id=DEFAULT_MODEL,
        released="2026-10-01",
        status="active",
        context_window=131_072,
        notes=(
            "Production default. Free OpenRouter tier; the free suffix is part "
            "of the id, so a paid route is a different pin."
        ),
    ),
    "nvidia/nemotron-3-ultra-550b-a55b": ModelPin(
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        released="2026-10-01",
        status="active",
        context_window=131_072,
        notes="Paid route for the same model; use when the free tier is rate-limited.",
    ),
    "anthropic/claude-sonnet-4.5": ModelPin(
        model_id="anthropic/claude-sonnet-4.5",
        released="2025-09-29",
        status="active",
        context_window=200_000,
        notes="Strong-model fallback for editorial and audit quality.",
    ),
    "openai/gpt-5-mini": ModelPin(
        model_id="openai/gpt-5-mini",
        released="2025-08-07",
        status="active",
        context_window=400_000,
        notes="Fast/cheap fallback for research tracks.",
    ),
    "google/gemini-2.5-flash": ModelPin(
        model_id="google/gemini-2.5-flash",
        released="2025-06-17",
        status="active",
        context_window=1_000_000,
        notes="Long-context fallback for large dossiers.",
    ),
}


def model_pin(model_id: str) -> ModelPin | None:
    """Return the registry entry for ``model_id``, or ``None`` if unpinned."""
    return MODEL_REGISTRY.get(model_id)


def validate_model(
    model_id: str,
    *,
    role: str = "model",
    allow_deprecated: bool = True,
) -> list[str]:
    """Validate one model id.

    Returns a list of human-readable warnings.  Raises
    :class:`ModelRegistryError` for a retired model, or for a deprecated model
    when ``allow_deprecated`` is false.  An unpinned model is a warning rather
    than an error so custom OpenRouter routes remain possible.
    """
    pin = MODEL_REGISTRY.get(model_id)
    if pin is None:
        return [
            f"{role} {model_id!r} is not pinned in the model registry; "
            "add a ModelPin before a production release"
        ]
    if pin.status == "retired":
        replacement = f"; migrate to {pin.replacement!r}" if pin.replacement else ""
        raise ModelRegistryError(f"{role} {model_id!r} is retired{replacement}")
    if pin.status == "deprecated":
        if not allow_deprecated:
            replacement = f"; migrate to {pin.replacement!r}" if pin.replacement else ""
            raise ModelRegistryError(f"{role} {model_id!r} is deprecated{replacement}")
        replacement = f"; migrate to {pin.replacement!r}" if pin.replacement else ""
        return [f"{role} {model_id!r} is deprecated{replacement}"]
    return []


def resolve_model_chain(
    primary: str,
    fallbacks: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return the usable model chain, dropping retired entries.

    This is the explicit fallback policy: the primary is tried first, then the
    configured fallbacks in order.  A retired model is never silently used.
    """
    chain: list[str] = []
    for model_id in (primary, *fallbacks):
        pin = MODEL_REGISTRY.get(model_id)
        if pin is not None and pin.status == "retired":
            continue
        if model_id not in chain:
            chain.append(model_id)
    if not chain:
        raise ModelRegistryError(
            f"no usable model in chain {primary!r} + {list(fallbacks)!r}; "
            "all entries are retired"
        )
    return tuple(chain)


def model_manifest(
    models: dict[str, str],
    *,
    fallbacks: tuple[str, ...] = (),
) -> dict[str, object]:
    """Serialise the exact pins used by a run for its manifest."""
    pins: dict[str, object] = {}
    for role, model_id in models.items():
        pin = MODEL_REGISTRY.get(model_id)
        pins[role] = (
            pin.as_dict()
            if pin is not None
            else {"model_id": model_id, "provider": "openrouter", "status": "unpinned"}
        )
    return {
        "pins": pins,
        "fallbacks": list(fallbacks),
        "chain": list(resolve_model_chain(models.get("strong", DEFAULT_MODEL), fallbacks)),
    }
