"""P2: deterministic entailment checks against archived source text."""

from __future__ import annotations

from mars.archive import EvidenceArchive
from mars.entailment import (
    audit_claims,
    audit_text,
    classify_score,
    support_score,
    verify_claim,
)
from mars.evidence import EvidenceStore


def _store(tmp_path, *sources: tuple[str, str]) -> EvidenceStore:
    archive = EvidenceArchive(tmp_path)
    store = EvidenceStore(archive=archive)
    for url, body in sources:
        store.register(url, url, body[:100], content=body, kind="web")
    return store


def test_support_score_rewards_matching_text() -> None:
    claim = "Agents use tools to accomplish goals beyond a single completion."
    source = "Agents use tools to accomplish goals beyond a single completion."

    assert support_score(claim, source) >= 0.9
    assert support_score(claim, "Bananas are rich in potassium.") < 0.2


def test_missing_numbers_are_penalised() -> None:
    claim = "Revenue grew 42% in 2024 for the agents division."
    source = "Revenue grew for the agents division."

    assert support_score(claim, source) < 0.55


def test_classify_score_thresholds() -> None:
    assert classify_score(0.8) == "supported"
    assert classify_score(0.4) == "partial"
    assert classify_score(0.1) == "unsupported"


def test_audit_text_checks_each_cited_sentence(tmp_path) -> None:
    store = _store(
        tmp_path,
        (
            "https://example.com/agents",
            "Agents use tools to accomplish goals beyond a single completion.",
        ),
    )
    draft = (
        "# Report\n\n"
        "Agents use tools to accomplish goals beyond a single completion. [S1]\n\n"
        "A sentence with no citation at all is recorded as unverified.\n"
    )

    report = audit_text(store, draft)

    assert report.total == 2
    assert report.supported == 1
    assert report.unverified == 1
    assert report.ratio == 0.5
    assert report.verdicts[1].notes == "no citation attached"


def test_audit_text_marks_unrelated_citation_unsupported(tmp_path) -> None:
    store = _store(
        tmp_path,
        ("https://example.com/a", "Bananas are rich in potassium and fibre."),
    )
    draft = "Agents use tools to accomplish goals beyond a single completion. [S1]"

    report = audit_text(store, draft)

    assert report.unsupported == 1
    assert report.ratio == 0.0
    assert report.unsupported_claims


def test_audit_claims_updates_the_claim_ledger(tmp_path) -> None:
    store = _store(
        tmp_path,
        (
            "https://example.com/agents",
            "Agents use tools to accomplish goals beyond a single completion.",
        ),
    )
    supported = store.add_claim(
        "Agents use tools to accomplish goals beyond a single completion.",
        ["S1"],
    )
    unsupported = store.add_claim("Bananas are rich in potassium.", ["S1"])

    report = audit_claims(store, store.claims())

    assert report.total == 2
    assert report.supported == 1
    assert report.unsupported == 1
    assert supported.verification == "supported"
    assert supported.verification_score > 0
    assert supported.verified_at
    assert unsupported.verification == "unsupported"


def test_verify_claim_without_citations_is_unverified(tmp_path) -> None:
    store = _store(tmp_path, ("https://example.com/a", "Some source text."))
    claim = store.add_claim("A claim without any source attached.", [])

    verdict = verify_claim(claim, store)

    assert verdict.status == "unverified"
    assert verdict.notes == "no citation attached"
