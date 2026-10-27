"""Tests for the LLM providers.

The OpenRouter tests run against :func:`fake_openrouter`, a local server that
speaks the OpenAI chat-completions protocol.  That exercises the real client
stack - ``langchain-openrouter`` -> OpenRouter SDK -> HTTP -> response parsing -
without an API key or network access.
"""

from __future__ import annotations

import json

import pytest

from mars.llm import OpenRouterLLM, ToolCall, build_llm
from mars.metrics import UsageTracker
from tests.support.fake_llm import MockLLM


def test_build_llm_always_returns_the_openrouter_client(settings) -> None:
    """Production has exactly one provider; there is no mock/offline fallback."""
    llm = build_llm(settings, "fast")
    assert isinstance(llm, OpenRouterLLM)
    # The live provider drives CrewAI's native tool loop.
    assert llm.supports_function_calling() is True


def test_build_llm_returns_an_openrouter_client_for_live_runs(settings) -> None:
    live = settings.with_overrides(llm_provider="openrouter", openrouter_api_key="sk-test")
    llm = build_llm(live, "strong")
    assert isinstance(llm, OpenRouterLLM)
    assert llm.supports_function_calling() is True
    assert llm.model == live.model_strong


def test_mock_llm_detects_roles_from_the_task_templates() -> None:
    mock = MockLLM()

    assert '"tracks"' in mock.reply("You are the research director for Mars", "go")
    assert '"findings"' in mock.reply("convert it into structured findings", "dossier")
    assert mock.reply(
        "Write the **foundations** section of an Agentic AI research report", "notes"
    ).startswith("## ")
    assert mock.reply(
        "Assemble the final Agentic AI report on agents", "sections"
    ).startswith("# ")
    assert '"verdict"' in mock.reply(
        "audit the draft report below against the source registry", "draft"
    )


def test_mock_llm_cites_only_registered_sources(context) -> None:
    context.evidence.register("https://example.com/one", "First source")
    mock = MockLLM(context=context)
    dossier = mock.reply(
        "Research the **foundations** track of an Agentic AI report on agentic AI",
        "notes",
    )

    assert "[S1]" in dossier
    assert "[S9]" not in dossier


def _tool_schema(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"The {name} tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
        for name in names
    ]


def test_mock_llm_asks_for_a_tool_before_it_answers(context) -> None:
    """The offline provider drives CrewAI's native tool loop."""
    mock = MockLLM(context=context)
    tools = _tool_schema("query_corpus", "web_search")

    first = mock.call(
        [{"role": "user", "content": "Research the **foundations** track of a report."}],
        tools=tools,
    )

    assert isinstance(first, list)
    assert first[0].function.name == "query_corpus"
    assert "query" in first[0].function.arguments


def test_mock_llm_uses_each_tool_once_and_then_answers(context) -> None:
    mock = MockLLM(context=context)
    tools = _tool_schema("query_corpus", "web_search")
    conversation: list[dict[str, str]] = [
        {"role": "user", "content": "Research the **foundations** track of a report."}
    ]

    seen: list[str] = []
    for _ in range(4):
        answer = mock.call(conversation, tools=tools)
        if not isinstance(answer, list):
            break
        call = answer[0]
        seen.append(call.function.name)
        conversation.append({"role": "assistant", "content": ""})
        conversation.append(
            {"role": "tool", "name": call.function.name, "content": "[S1] A source"}
        )

    assert seen == ["query_corpus", "web_search"]
    assert isinstance(mock.call(conversation, tools=tools), str)


def test_mock_llm_does_not_call_tools_for_non_research_roles(context) -> None:
    mock = MockLLM(context=context)
    tools = _tool_schema("query_corpus")

    answer = mock.call(
        [{"role": "user", "content": "Assemble the final Agentic AI report on agentic AI."}],
        tools=tools,
    )

    assert isinstance(answer, str)


def test_mock_llm_detects_the_track_from_the_task_template() -> None:
    mock = MockLLM()

    assert (
        mock._detect_track("Write the **capabilities** section of a report.")
        == "capabilities"
    )
    assert mock._detect_track("Research the **future** track of a report.") == "future"
    assert (
        mock._detect_track('{"track": "foundations", "findings": []}')
        == "foundations"
    )
    # Keyword counting must not be fooled by an unrelated era mentioned in a
    # source title.
    assert (
        mock._detect_track("The **capabilities** track of a report mentions history.")
        == "capabilities"
    )


def test_mock_llm_writes_a_section_from_the_findings_it_was_given(context) -> None:
    from mars.agents.tasks import writer_task

    context.evidence.register("https://example.com/one", "First source")
    findings = (
        '{"track": "capabilities", "findings": ['
        '{"text": "Agents are already deployed in production.", '
        '"evidence": ["S1"], "confidence": 0.8, "kind": "fact"}]}'
    )
    description, _expected = writer_task(
        "agentic AI", "capabilities", "## Findings\n- Something [S1]", findings
    )

    mock = MockLLM(context=context)
    section = mock.reply(description, description)

    assert section.startswith("## Capabilities, Benchmarks and Evaluation")
    assert "Agents are already deployed in production." in section
    assert "[S1]" in section
    assert "foundations" not in section.split("### What this means")[0]


def test_mock_llm_auditor_measures_the_draft_it_is_given(context) -> None:
    from mars.agents.tasks import auditor_task

    context.evidence.register("https://example.com/one", "First source")
    draft = (
        "# Report\n\n"
        "Agents are already deployed in production systems today, across support, "
        "research and operations teams in a growing number of large organisations. [S1]\n\n"
        "Another claim that makes an assertion without any supporting source at all, "
        "and which runs on past the minimum length that the auditor counts as prose.\n"
    )
    description, _expected = auditor_task(draft, "[S1] First source")

    mock = MockLLM(context=context)
    audit = json.loads(mock.reply(description, description))

    assert audit["grounded_ratio"] == pytest.approx(0.5)
    assert audit["total_claims"] == 2
    assert audit["cited_claims"] == 1
    assert audit["verdict"] == "revise"
    assert audit["uncited_paragraphs"]


def test_openrouter_llm_returns_text_for_a_plain_call(fake_openrouter) -> None:
    usage = UsageTracker()
    llm = OpenRouterLLM(
        model="test/model",
        api_key="sk-test",
        base_url=fake_openrouter,
        usage=usage,
        cache_enabled=False,
    )

    answer = llm.call(
        [
            {"role": "system", "content": "You are the research director for Mars"},
            {"role": "user", "content": "plan it"},
        ]
    )

    assert isinstance(answer, str)
    assert '"tracks"' in answer
    assert usage.calls == 1


def test_openrouter_llm_returns_crewai_style_tool_calls(fake_openrouter) -> None:
    llm = OpenRouterLLM(
        model="test/model",
        api_key="sk-test",
        base_url=fake_openrouter,
        cache_enabled=False,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_corpus",
                "description": "Search the curated corpus.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    result = llm.call([{"role": "user", "content": "search"}], tools=tools)

    assert isinstance(result, list)
    assert result
    call = result[0]
    assert isinstance(call, ToolCall)
    assert call.function.name == "query_corpus"
    assert "query" in call.function.arguments


def test_openrouter_llm_serves_cached_answers(fake_openrouter, tmp_path) -> None:
    llm = OpenRouterLLM(
        model="test/model",
        api_key="sk-test",
        base_url=fake_openrouter,
        cache_dir=tmp_path / "cache",
        cache_enabled=True,
    )
    messages = [{"role": "user", "content": "remember this"}]

    first = llm.call(messages)
    second = llm.call(messages)

    assert first == second
    assert llm.usage.cached_calls == 1


def test_openrouter_llm_returns_validated_structured_output(fake_openrouter) -> None:
    """The structured path (used by the director, analyst and auditor)."""
    from mars.schemas import ResearchPlan

    llm = OpenRouterLLM(
        model="test/model",
        api_key="sk-test",
        base_url=fake_openrouter,
        cache_enabled=False,
    )

    plan = llm.call(
        [
            {"role": "system", "content": "You are the research director for Mars"},
            {"role": "user", "content": "plan it"},
        ],
        response_model=ResearchPlan,
    )

    assert isinstance(plan, ResearchPlan)
    assert plan.topic
    # Structured calls must still count against the run's token/cost ceilings.
    assert llm.usage.prompt_tokens > 0
    assert llm.usage.completion_tokens > 0
    assert {track.track for track in plan.tracks} == {
        "foundations",
        "architectures",
        "capabilities",
        "applications",
        "safety",
        "future",
    }


def test_openrouter_llm_hands_the_timeout_to_the_client_in_milliseconds(
    fake_openrouter,
) -> None:
    """``langchain-openrouter`` reads the timeout as milliseconds.

    Passing seconds straight through silently turns a two-minute budget into a
    120ms one, which looks exactly like the provider hanging.
    """
    llm = OpenRouterLLM(
        model="test/model",
        api_key="sk-test",
        base_url=fake_openrouter,
        request_timeout=12.5,
        cache_enabled=False,
    )

    client = llm._ensure_client()

    assert client.request_timeout == 12_500


def test_openrouter_llm_disables_sdk_retries(fake_openrouter) -> None:
    """Mars owns retries so they share a circuit breaker and hard budget."""
    llm = OpenRouterLLM(
        model="test/model",
        api_key="sk-test",
        base_url=fake_openrouter,
        max_retries=9,
        cache_enabled=False,
    )

    client = llm._ensure_client()

    assert client.max_retries == 0
