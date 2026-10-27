"""Deterministic claim-level entailment checks.

The production audit combines two signals:

* an LLM auditor that judges whether a sentence is supported, and
* this module, which checks each cited sentence against the archived text of
  the sources it cites.

The lexical check is deliberately conservative.  It catches the common failure
mode the old grounding score missed - a sentence that carries ``[S3]`` but
whose content is nowhere in ``S3`` - without pretending to be a full natural
language inference model.  The LLM audit supplies semantic judgement; this
module supplies a reproducible floor and a per-claim record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ClaimVerdict",
    "EntailmentReport",
    "support_score",
    "classify_score",
    "split_sentences",
    "verify_claim",
    "audit_text",
    "audit_claims",
]

_CITATION_RE = re.compile(r"\[(S\d+)\]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\u2019\-]{2,}")
_NUMBER_RE = re.compile(r"\b\d{4}\b|\b\d+(?:[.,]\d+)?%?\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")

_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does for from had
    has have how i if in into is it its may might more most must no not of on
    or our out over said say says she should so some such than that the their
    them then there these they this those to too under up us was we were what
    when where which while who why will with would you your
    """.split()
)

# A sentence has to be long enough to be a factual assertion rather than a
# label, heading or transition.
_MIN_CLAIM_WORDS = 5


@dataclass(slots=True)
class ClaimVerdict:
    """The result of checking one sentence or ledger claim."""

    text: str
    citations: list[str] = field(default_factory=list)
    status: str = "unverified"  # supported | partial | unsupported | unverified
    score: float = 0.0
    notes: str = ""

    @property
    def supported(self) -> bool:
        return self.status == "supported"


@dataclass(slots=True)
class EntailmentReport:
    """Aggregate entailment result for a body of text or a claim ledger."""

    ratio: float = 0.0
    total: int = 0
    supported: int = 0
    partial: int = 0
    unsupported: int = 0
    unverified: int = 0
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def unsupported_claims(self) -> list[str]:
        return [verdict.text[:300] for verdict in self.verdicts if verdict.status == "unsupported"]

    @property
    def unverified_claims(self) -> list[str]:
        return [verdict.text[:300] for verdict in self.verdicts if verdict.status == "unverified"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ratio": round(self.ratio, 4),
            "total": self.total,
            "supported": self.supported,
            "partial": self.partial,
            "unsupported": self.unsupported,
            "unverified": self.unverified,
            "claims": [
                {
                    "text": verdict.text[:500],
                    "citations": verdict.citations,
                    "status": verdict.status,
                    "score": round(verdict.score, 4),
                    "notes": verdict.notes,
                }
                for verdict in self.verdicts
            ],
        }


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if token.lower() not in _STOPWORDS
    }


def _numbers(text: str) -> set[str]:
    return {value.replace(",", "") for value in _NUMBER_RE.findall(text or "")}


def support_score(claim: str, source_text: str) -> float:
    """Lexical entailment score in ``[0, 1]`` for a claim against one source.

    The score combines term coverage with a number/date check.  A claim that
    says "revenue grew 42% in 2024" is not supported by a source that only
    mentions "revenue grew": missing concrete values are penalised.
    """
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return 0.0
    source_tokens = _tokens(source_text)
    if not source_tokens:
        return 0.0
    coverage = len(claim_tokens & source_tokens) / len(claim_tokens)

    claim_numbers = _numbers(claim)
    if claim_numbers:
        number_factor = len(claim_numbers & _numbers(source_text)) / len(claim_numbers)
    else:
        number_factor = None

    # Named entities / proper nouns are a useful proxy for specificity.  If a
    # claim names three organisations and the source names none of them, the
    # coverage number alone is too generous.
    claim_names = {
        token
        for token in re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", claim or "")
        if token.lower() not in _STOPWORDS
    }
    if claim_names:
        name_factor = len({name.lower() for name in claim_names} & source_tokens) / len(claim_names)
    else:
        name_factor = 1.0

    if number_factor is None:
        score = 0.85 * coverage + 0.15 * name_factor
    else:
        score = 0.6 * coverage + 0.25 * number_factor + 0.15 * name_factor
        # A claim that states concrete values the source never mentions cannot
        # be treated as fully supported, however well the prose overlaps.
        if number_factor < 1.0:
            score = min(score, 0.5)
    return max(0.0, min(1.0, score))


def classify_score(
    score: float,
    *,
    supported_at: float = 0.55,
    partial_at: float = 0.30,
) -> str:
    if score >= supported_at:
        return "supported"
    if score >= partial_at:
        return "partial"
    return "unsupported"


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences while preserving citation markers."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT.split(cleaned)
    return [part.strip() for part in parts if part.strip()]


def _source_texts(store: Any, citations: list[str]) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    for citation in citations:
        source = store.get(citation)
        if source is None:
            continue
        archive = getattr(store, "archive", None)
        text = archive.resolve(source) if archive is not None else None
        if not text:
            text = getattr(source, "snippet", "") or ""
        if text:
            texts.append((citation, text))
    return texts


def _verdict_for(
    text: str,
    citations: list[str],
    store: Any,
    *,
    supported_at: float,
    partial_at: float,
) -> ClaimVerdict:
    clean = _CITATION_RE.sub(" ", text or "").strip()
    if len(clean.split()) < _MIN_CLAIM_WORDS:
        return ClaimVerdict(text=clean, citations=citations, status="unverified", notes="too short")
    if not citations:
        return ClaimVerdict(
            text=clean,
            citations=[],
            status="unverified",
            notes="no citation attached",
        )

    texts = _source_texts(store, citations)
    if not texts:
        return ClaimVerdict(
            text=clean,
            citations=citations,
            status="unverified",
            notes="cited source has no archived text or snippet",
        )

    best_score = 0.0
    best_id = ""
    for citation, source_text in texts:
        score = support_score(clean, source_text)
        if score > best_score:
            best_score = score
            best_id = citation
    status = classify_score(best_score, supported_at=supported_at, partial_at=partial_at)
    return ClaimVerdict(
        text=clean,
        citations=citations,
        status=status,
        score=best_score,
        notes=f"best match {best_id or 'n/a'}",
    )


def verify_claim(
    claim: Any,
    store: Any,
    *,
    supported_at: float = 0.55,
    partial_at: float = 0.30,
) -> ClaimVerdict:
    """Verify one :class:`~mars.schemas.Claim` against its cited sources."""
    citations = [str(cid).upper() for cid in (getattr(claim, "evidence", []) or [])]
    return _verdict_for(
        getattr(claim, "text", "") or "",
        citations,
        store,
        supported_at=supported_at,
        partial_at=partial_at,
    )


def audit_text(
    store: Any,
    text: str,
    *,
    supported_at: float = 0.55,
    partial_at: float = 0.30,
) -> EntailmentReport:
    """Check every substantive sentence in a report body."""
    verdicts: list[ClaimVerdict] = []
    for block in (text or "").split("\n\n"):
        stripped = block.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            continue
        if stripped.startswith("```"):
            continue
        for sentence in split_sentences(stripped):
            if len(sentence.split()) < _MIN_CLAIM_WORDS:
                continue
            citations = [cid.upper() for cid in _CITATION_RE.findall(sentence)]
            verdicts.append(
                _verdict_for(
                    sentence,
                    citations,
                    store,
                    supported_at=supported_at,
                    partial_at=partial_at,
                )
            )
    return _aggregate(verdicts)


def audit_claims(
    store: Any,
    claims: list[Any] | None = None,
    *,
    supported_at: float = 0.55,
    partial_at: float = 0.30,
) -> EntailmentReport:
    """Verify the persisted claim ledger and update each claim's status."""
    values = claims if claims is not None else store.claims()
    verdicts: list[ClaimVerdict] = []
    for claim in values:
        verdict = verify_claim(
            claim, store, supported_at=supported_at, partial_at=partial_at
        )
        try:
            claim.verification = verdict.status  # type: ignore[attr-defined]
            claim.verification_score = round(verdict.score, 4)  # type: ignore[attr-defined]
            claim.verification_notes = verdict.notes  # type: ignore[attr-defined]
            from .schemas import utcnow_iso

            claim.verified_at = utcnow_iso()  # type: ignore[attr-defined]
        except Exception:
            pass
        verdicts.append(verdict)
    return _aggregate(verdicts)


def _aggregate(verdicts: list[ClaimVerdict]) -> EntailmentReport:
    total = len(verdicts)
    supported = sum(1 for verdict in verdicts if verdict.status == "supported")
    partial = sum(1 for verdict in verdicts if verdict.status == "partial")
    unsupported = sum(1 for verdict in verdicts if verdict.status == "unsupported")
    unverified = sum(1 for verdict in verdicts if verdict.status == "unverified")
    ratio = (supported + 0.5 * partial) / total if total else 0.0
    return EntailmentReport(
        ratio=round(ratio, 4),
        total=total,
        supported=supported,
        partial=partial,
        unsupported=unsupported,
        unverified=unverified,
        verdicts=verdicts,
    )
