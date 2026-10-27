"""Tests for the pinned model registry and its lifecycle policy."""

from __future__ import annotations

import pytest

from mars.config import build_settings
from mars.model_registry import (
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    ModelPin,
    ModelRegistryError,
    model_manifest,
    model_pin,
    resolve_model_chain,
    validate_model,
)


def test_default_model_is_pinned_and_active() -> None:
    pin = model_pin(DEFAULT_MODEL)
    assert pin is not None
    assert pin.status == "active"
    assert pin.context_window


def test_unpinned_model_is_a_warning_not_an_error() -> None:
    warnings = validate_model("vendor/unknown-model")
    assert len(warnings) == 1
    assert "not pinned" in warnings[0]


def test_retired_model_is_a_hard_error(monkeypatch) -> None:
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "vendor/retired-model",
        ModelPin(
            model_id="vendor/retired-model",
            status="retired",
            replacement="vendor/new-model",
        ),
    )
    with pytest.raises(ModelRegistryError, match="retired"):
        validate_model("vendor/retired-model")


def test_deprecated_model_warns_or_errors(monkeypatch) -> None:
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "vendor/deprecated-model",
        ModelPin(
            model_id="vendor/deprecated-model",
            status="deprecated",
            replacement="vendor/new-model",
        ),
    )
    warnings = validate_model("vendor/deprecated-model")
    assert warnings and "deprecated" in warnings[0]
    with pytest.raises(ModelRegistryError, match="deprecated"):
        validate_model("vendor/deprecated-model", allow_deprecated=False)


def test_fallback_chain_drops_retired_and_deduplicates(monkeypatch) -> None:
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "vendor/retired-model",
        ModelPin(model_id="vendor/retired-model", status="retired"),
    )
    chain = resolve_model_chain(
        DEFAULT_MODEL, ("vendor/retired-model", DEFAULT_MODEL, "vendor/other")
    )
    assert chain == (DEFAULT_MODEL, "vendor/other")


def test_fallback_chain_with_only_retired_models_raises(monkeypatch) -> None:
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "vendor/retired-model",
        ModelPin(model_id="vendor/retired-model", status="retired"),
    )
    with pytest.raises(ModelRegistryError, match="no usable model"):
        resolve_model_chain("vendor/retired-model")


def test_model_manifest_records_pins_and_chain() -> None:
    manifest = model_manifest(
        {"strong": DEFAULT_MODEL, "fast": "vendor/unknown"},
        fallbacks=("vendor/other",),
    )
    assert manifest["pins"]["strong"]["model_id"] == DEFAULT_MODEL
    assert manifest["pins"]["strong"]["status"] == "active"
    assert manifest["pins"]["fast"]["status"] == "unpinned"
    assert manifest["fallbacks"] == ["vendor/other"]
    assert manifest["chain"] == [DEFAULT_MODEL, "vendor/other"]


def test_retired_primary_with_active_fallback_is_allowed(monkeypatch) -> None:
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "vendor/retired-model",
        ModelPin(model_id="vendor/retired-model", status="retired"),
    )
    settings = build_settings(env_file=None).with_overrides(
        openrouter_api_key="sk-test",
        model_strong="vendor/retired-model",
        model_fallbacks=(DEFAULT_MODEL,),
    )
    settings.validate()
    assert any("fallback" in warning for warning in settings.warnings())


def test_retired_fallback_entries_are_dropped_not_fatal(monkeypatch) -> None:
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "vendor/retired-model",
        ModelPin(model_id="vendor/retired-model", status="retired"),
    )
    settings = build_settings(env_file=None).with_overrides(
        openrouter_api_key="sk-test",
        model_fallbacks=("vendor/retired-model", DEFAULT_MODEL),
    )
    settings.validate()


def test_pipeline_resolves_a_retired_primary_before_running(settings, monkeypatch) -> None:
    from mars.pipeline import ResearchPipeline

    monkeypatch.setitem(
        MODEL_REGISTRY,
        "vendor/retired-model",
        ModelPin(model_id="vendor/retired-model", status="retired"),
    )
    configured = settings.with_overrides(
        model_strong="vendor/retired-model",
        model_fallbacks=(DEFAULT_MODEL,),
    )
    result = ResearchPipeline(configured).run()
    assert result.report.metrics["models"]["pins"]["strong"]["model_id"] == DEFAULT_MODEL
    assert result.report.metrics["models"]["fallbacks"] == [DEFAULT_MODEL]


def test_settings_reject_a_retired_model(monkeypatch) -> None:
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "vendor/retired-model",
        ModelPin(model_id="vendor/retired-model", status="retired"),
    )
    settings = build_settings(env_file=None).with_overrides(
        openrouter_api_key="sk-test", model_strong="vendor/retired-model"
    )
    with pytest.raises(Exception, match="retired"):
        settings.validate()


def test_strict_registry_rejects_unpinned_models() -> None:
    settings = build_settings(env_file=None).with_overrides(
        openrouter_api_key="sk-test",
        model_strong="vendor/unknown",
        strict_model_registry=True,
    )
    with pytest.raises(Exception, match="not pinned"):
        settings.validate()


def test_non_strict_registry_accepts_unpinned_models() -> None:
    settings = build_settings(env_file=None).with_overrides(
        openrouter_api_key="sk-test", model_strong="vendor/unknown"
    )
    settings.validate()
    assert any("not pinned" in warning for warning in settings.warnings())
