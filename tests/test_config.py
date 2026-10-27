"""Tests for configuration, environment loading and depth presets."""

from __future__ import annotations

import os

from mars.config import TRACKS, Settings, build_settings, load_dotenv


def test_load_dotenv_parses_common_shapes(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MARS_TOPIC", raising=False)
    monkeypatch.delenv("MARS_DEPTH", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "MARS_TOPIC='agentic AI'",
                "export MARS_DEPTH=deep",
                "OPENROUTER_API_KEY=sk-test",
                "not a pair",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_dotenv(env_file)

    assert loaded["MARS_TOPIC"] == "agentic AI"
    assert loaded["MARS_DEPTH"] == "deep"
    assert loaded["OPENROUTER_API_KEY"] == "sk-test"
    assert "not a pair" not in loaded
    assert os.environ["MARS_TOPIC"] == "agentic AI"


def test_load_dotenv_does_not_override_existing_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MARS_TOPIC", "already-set")
    env_file = tmp_path / ".env"
    env_file.write_text("MARS_TOPIC=from-file\n", encoding="utf-8")

    load_dotenv(env_file)

    assert os.environ["MARS_TOPIC"] == "already-set"


def test_load_dotenv_missing_file_is_a_no_op(tmp_path) -> None:
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_depth_presets_change_budgets(tmp_path) -> None:
    quick = build_settings(depth="quick", env_file=None, output_dir=tmp_path)
    deep = build_settings(depth="deep", env_file=None, output_dir=tmp_path)
    standard = build_settings(depth="standard", env_file=None, output_dir=tmp_path)

    assert quick.max_rounds == 1
    assert quick.max_search_results == 5
    assert deep.max_rounds == 3
    assert deep.max_search_results == 12
    assert deep.max_tool_iterations > standard.max_tool_iterations


def test_with_overrides_ignores_none(tmp_path) -> None:
    settings = Settings(topic="x")
    updated = settings.with_overrides(topic="y", depth=None)
    assert updated.topic == "y"
    assert updated.depth == settings.depth
    assert settings.topic == "x"  # the original is untouched


def test_has_live_llm_requires_a_key() -> None:
    assert not Settings(llm_provider="openrouter").has_live_llm
    assert not Settings(llm_provider="openrouter", openrouter_api_key="").has_live_llm
    assert Settings(llm_provider="openrouter", openrouter_api_key="sk-x").has_live_llm
    assert not Settings(llm_provider="other", openrouter_api_key="sk-x").has_live_llm


def test_build_settings_reads_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MARS_TOPIC", "llm agents")
    monkeypatch.setenv("MARS_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MARS_OUTPUT_DIR", str(tmp_path / "out"))

    settings = build_settings(env_file=None)

    assert settings.topic == "llm agents"
    assert settings.llm_provider == "openrouter"
    assert settings.output_dir == tmp_path / "out"


def test_to_dict_is_json_friendly(tmp_path) -> None:
    settings = build_settings(env_file=None, output_dir=tmp_path / "out")
    data = settings.to_dict()
    assert isinstance(data["cache_dir"], str)
    assert isinstance(data["output_dir"], str)
    assert list(data["tracks"]) == list(TRACKS)
