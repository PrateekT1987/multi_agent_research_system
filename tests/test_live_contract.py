"""Opt-in live provider contract test.

Run explicitly with::

    OPENROUTER_API_KEY=... python -m pytest -m live -q

The default test run deselects ``live`` markers, so CI never spends money or
depends on an external service.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


def test_configured_openrouter_model_answers() -> None:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        pytest.skip("OPENROUTER_API_KEY is not set")

    from mars.config import build_settings
    from mars.llm import build_llm

    settings = build_settings(
        llm_provider="openrouter",
        openrouter_api_key=key,
        model_fast="nvidia/nemotron-3-ultra-550b-a55b",
        model_strong="nvidia/nemotron-3-ultra-550b-a55b",
        model_planner="nvidia/nemotron-3-ultra-550b-a55b",
        cache_enabled=False,
        env_file=None,
    )
    settings.validate()
    llm = build_llm(settings, "fast")

    reply = llm.call("ping")

    assert isinstance(reply, str)
    assert reply.strip()
