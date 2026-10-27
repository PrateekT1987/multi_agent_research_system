"""Tests for the evidence store."""

from __future__ import annotations

from mars.evidence import EvidenceStore, normalise_url


def test_register_is_idempotent_per_url() -> None:
    store = EvidenceStore()
    first = store.register("https://example.com/a/", "Title A", "snippet")
    second = store.register("https://www.example.com/a", "Title A", "longer snippet")

    assert first.id == second.id
    assert len(store.sources()) == 1
    assert second.snippet == "longer snippet"


def test_normalise_url_strips_tracking_parameters() -> None:
    assert normalise_url("https://Example.com/path/?utm_source=x&id=3#frag") == (
        "https://example.com/path?id=3"
    )


def test_citations_are_extracted_in_order() -> None:
    store = EvidenceStore()
    store.register("https://example.com/1", "One")
    store.register("https://example.com/2", "Two")
    text = "First [S2], then [S1], and again [S2]."
    assert store.extract_citations(text) == ["S2", "S1"]


def test_dangling_citations_are_detected() -> None:
    store = EvidenceStore()
    store.register("https://example.com/1", "One")
    assert store.dangling_citations("claim [S1] and [S9]") == ["S9"]


def test_grounding_score_counts_cited_paragraphs() -> None:
    store = EvidenceStore()
    store.register("https://example.com/1", "One")
    text = (
        "A paragraph with a citation and quite a lot of words in it. [S1]\n\n"
        "A paragraph without any citation at all, but long enough to count as prose.\n\n"
        "## A heading is ignored\n\n"
        "Short. [S1]"
    )
    ratio, cited, total = store.grounding_score(text)
    assert (cited, total) == (1, 2)
    assert ratio == 0.5


def test_uncited_paragraphs_are_flagged() -> None:
    store = EvidenceStore()
    text = "A long paragraph that makes a claim without any citation whatsoever here."
    flagged = store.uncited_paragraphs(text)
    assert flagged and flagged[0].startswith("A long paragraph")


def test_add_claim_records_evidence() -> None:
    store = EvidenceStore()
    store.register("https://example.com/1", "One")
    claim = store.add_claim("Agents use tools.", ["s1"], confidence=0.8, track="current")
    assert claim.evidence == ["S1"]
    assert store.claims("current") == [claim]
    assert store.claims("history") == []


def test_reference_list_renders_markdown() -> None:
    store = EvidenceStore()
    store.register("https://example.com/1", "One", "snippet")
    rendered = store.render_reference_list()
    assert "[S1]" in rendered
    assert "https://example.com/1" in rendered
