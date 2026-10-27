"""Hermetic regression tests for the P0 evidence-integrity gates.

The run these tests encode produced a finished-looking report from four
evidence-free tracks: the researcher's dossier was an unexecuted tool call, the
analyst returned nothing, the writer filled the gap from its own priors, and
the pipeline counted "the track did not raise" as coverage.

Each test here pins one gate that now stops that path:

* provider error envelopes are classified (and transient ones retried),
* an unusable dossier is rejected before analysis,
* an empty analyst result falls back to the claim ledger,
* a section with no resolvable citations never reaches the report,
* coverage is measured by evidence, not by absence of an exception,
* the auditor is shown exactly the sources the draft cites.
"""

from __future__ import annotations

import json

import pytest

import mars.pipeline as pipeline_module
from mars.agents import dossier_problem
from mars.config import Settings
from mars.llm import OpenRouterLLM
from mars.pipeline import ResearchPipeline, ResearchTrackIncomplete, coverage_requirement
from mars.reliability import ProviderResponseError, ProviderUnavailableError
from mars.schemas import FindingBundle


class _EnvelopeError(Exception):
    """Stand-in for the SDK's ``ResponseValidationError`` on a 200 envelope."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"response validation failed: {code}")
        self.body = json.dumps({"id": "gen-test", "error": {"code": code, "message": message}})


def _llm(**kwargs) -> OpenRouterLLM:
    return OpenRouterLLM(
        model="test/model",
        api_key="sk-test",
        max_retries=kwargs.pop("max_retries", 0),
        retry_base_seconds=0.0,
        **kwargs,
    )


# ---------------------------------------------------------------- provider
def test_provider_envelope_is_read_from_the_sdk_error_body() -> None:
    exc = _EnvelopeError("provider_overloaded", "upstream is saturated")

    assert OpenRouterLLM._provider_envelope(exc) == {
        "code": "provider_overloaded",
        "message": "upstream is saturated",
    }


def test_transient_envelope_is_retried_and_then_succeeds() -> None:
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _EnvelopeError("provider_overloaded")
        return "ok"

    assert _llm(max_retries=1)._invoke_with_resilience(operation) == "ok"
    assert calls["n"] == 2


def test_transient_envelope_becomes_a_typed_provider_error() -> None:
    llm = _llm()

    with pytest.raises(ProviderResponseError) as caught:
        llm._invoke_with_resilience(lambda: (_ for _ in ()).throw(_EnvelopeError("provider_overloaded")))

    assert caught.value.code == "provider_overloaded"


def test_hard_quota_envelope_is_not_retried() -> None:
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        raise _EnvelopeError("insufficient_quota")

    with pytest.raises(ProviderUnavailableError, match="hard quota"):
        _llm(max_retries=2)._invoke_with_resilience(operation)
    assert calls["n"] == 1


def test_unknown_envelope_is_fatal_not_retried() -> None:
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        raise _EnvelopeError("invalid_request_error")

    with pytest.raises(ProviderResponseError):
        _llm(max_retries=2)._invoke_with_resilience(operation)
    assert calls["n"] == 1


# ------------------------------------------------------------------ dossier
def test_dossier_gate_rejects_an_unexecuted_tool_call() -> None:
    stub = (
        '<tool_call><function=web_search><parameter=query>agentic AI'
        "</parameter></function></tool_call>"
    )

    assert dossier_problem(stub)


def test_dossier_gate_rejects_uncited_prose() -> None:
    assert dossier_problem("word " * 80)


def test_dossier_gate_accepts_a_cited_dossier() -> None:
    dossier = "## Findings\n" + "- A sourced claim. [S1]\n" * 20

    assert dossier_problem(dossier) is None


# ------------------------------------------------------- pipeline fallbacks
def test_analyst_with_no_findings_falls_back_to_the_claim_ledger(
    settings: Settings, monkeypatch
) -> None:
    original = pipeline_module.run_research

    def research_and_record(context, llm, topic, track, plan=None, **kwargs):
        dossier = original(context, llm, topic, track, plan, **kwargs)
        ids = context.evidence.extract_citations(dossier)
        context.evidence.add_claim(
            f"Ledger claim for {track}.", ids[:1], track=track
        )
        return dossier

    def empty_analysis(llm, track, dossier, **kwargs):
        return FindingBundle(track=track, summary="nothing structured")

    monkeypatch.setattr(pipeline_module, "run_research", research_and_record)
    monkeypatch.setattr(pipeline_module, "run_analysis", empty_analysis)

    result = ResearchPipeline(settings).run()

    assert result.report.sections
    assert all(section.findings for section in result.report.sections)
    assert all(section.source_ids for section in result.report.sections)


def test_uncited_section_is_not_added_to_the_report(settings: Settings, monkeypatch) -> None:
    original = pipeline_module.run_writing

    def uncited_for_safety(*args, **kwargs):
        if args[2] == "safety":
            return "## Safety\n\nThis section asserts things with no sources at all."
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_writing", uncited_for_safety)

    result = ResearchPipeline(settings).run()

    safety = next(section for section in result.report.sections if section.track == "safety")
    assert safety.warnings
    assert safety.section_markdown == ""
    assert "## Tracks not covered" in result.report_markdown
    assert "Safety" in result.report_markdown
    # The surviving tracks still produced a report.
    assert result.report.audit.status == "degraded"


def test_coverage_gate_stops_a_run_with_too_few_evidence_tracks(
    settings: Settings, monkeypatch
) -> None:
    original = pipeline_module.run_research
    keep = {"foundations", "architectures"}

    def only_two(context, llm, topic, track, plan=None, **kwargs):
        if track not in keep:
            raise ResearchTrackIncomplete("the dossier cited no sources")
        return original(context, llm, topic, track, plan, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_research", only_two)

    planned = len(settings.tracks)
    required = coverage_requirement(planned)
    assert required > 2

    with pytest.raises(
        ProviderUnavailableError, match=f"only 2 of {planned} tracks"
    ):
        ResearchPipeline(settings).run()


def test_auditor_sees_every_source_the_draft_cites(settings: Settings, monkeypatch) -> None:
    original = pipeline_module.run_audit
    captured: list[str] = []

    def capture(llm, draft, sources_block, **kwargs):
        captured.append(sources_block)
        return original(llm, draft, sources_block, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_audit", capture)

    result = ResearchPipeline(settings).run()

    assert captured
    block = captured[-1]
    cited = result.evidence.extract_citations(result.report.draft_markdown)
    assert "REGISTRY SCOPE:" in block
    assert f"{len(cited)} of" in block
    assert "more." not in block
    for source_id in cited:
        assert f"[{source_id}]" in block
