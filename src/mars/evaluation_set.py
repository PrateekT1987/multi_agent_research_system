"""A fixed, versioned evaluation set and regression checks for Mars.

The unit tests prove the system is internally consistent.  They cannot prove
that a prompt change, a model migration or an agent refactor did not silently
make reports worse.  This module adds the missing piece: a small, curated set
of representative Agentic AI tasks (plus off-domain and injection probes) and
deterministic assertions that can be run against saved reports.

Two things are checked:

* **The gate cases** run without an LLM.  They assert that the domain lock
  still accepts in-domain topics and refuses off-domain ones, and that an
  injection-laden topic is still treated as a topic rather than an instruction.
* **The research cases** are evaluated against saved report JSON files.  They
  assert structure (sections, canonical tracks), evidence discipline
  (sources, findings, ``[S#]`` citations) and absence of forbidden content.

``compare_metrics`` adds the release check: a new run may not regress grounding
or entailment beyond a tolerance, and may not blow up cost.  The dataset lives
in ``evals/`` and is versioned with the code, so an intentional change is a
reviewable diff.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .domain import DOMAIN, DOMAIN_TRACK_KEYS, DomainError, ensure_domain_topic

__all__ = [
    "DEFAULT_EVAL_PATH",
    "CaseResult",
    "EvalCase",
    "EvalCheck",
    "EvaluationSet",
    "EvaluationSetError",
    "EvaluationSummary",
    "compare_metrics",
    "evaluate_case",
    "evaluate_dataset",
    "load_evaluation_set",
    "validate_evaluation_set",
]

DEFAULT_EVAL_PATH = Path("evals") / "agentic_ai_v1.json"

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_SECRET_MARKERS = (
    "openrouter_api_key",
    "sk-or-",
    "authorization: bearer",
    "-----begin private key-----",
)
_CASE_KINDS = ("research", "domain_gate", "injection")
_CASE_EXPECTATIONS = ("accept", "reject")


class EvaluationSetError(ValueError):
    """The evaluation dataset is missing, malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One regression case."""

    id: str
    topic: str
    kind: str = "research"
    expect: str = "accept"
    required_tracks: tuple[str, ...] = ()
    min_sections: int = 1
    min_sources: int = 1
    min_findings: int = 1
    min_grounding: float | None = None
    require_citations: bool = True
    forbid_terms: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: str = ""

    @property
    def needs_report(self) -> bool:
        return self.expect == "accept" and self.kind in {"research", "injection"}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSet:
    """A named, versioned collection of regression cases."""

    name: str
    version: str
    cases: tuple[EvalCase, ...]
    created: str = ""
    description: str = ""
    path: str | None = None

    def case(self, case_id: str) -> EvalCase | None:
        return next((case for case in self.cases if case.id == case_id), None)

    def by_topic(self, topic: str) -> EvalCase | None:
        normalised = _normalise_topic(topic)
        return next(
            (case for case in self.cases if _normalise_topic(case.topic) == normalised),
            None,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "created": self.created,
            "cases": len(self.cases),
            "path": self.path,
        }


@dataclass(frozen=True, slots=True)
class EvalCheck:
    """One assertion inside a case."""

    name: str
    ok: bool
    detail: str = ""


@dataclass(slots=True)
class CaseResult:
    """The outcome of one case."""

    case_id: str
    topic: str
    kind: str
    status: str  # pass | fail | skipped
    checks: list[EvalCheck] = field(default_factory=list)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"pass", "skipped"}


@dataclass(slots=True)
class EvaluationSummary:
    """Aggregate result for a whole dataset."""

    dataset: str
    version: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.status == "pass")

    @property
    def failed(self) -> int:
        return sum(1 for result in self.results if result.status == "fail")

    @property
    def skipped(self) -> int:
        return sum(1 for result in self.results if result.status == "skipped")

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "ok": self.ok,
            "results": [
                {
                    "case_id": result.case_id,
                    "topic": result.topic,
                    "kind": result.kind,
                    "status": result.status,
                    "message": result.message,
                    "checks": [asdict(check) for check in result.checks],
                }
                for result in self.results
            ],
        }


def _normalise_topic(topic: str) -> str:
    return " ".join((topic or "").lower().split())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationSetError(message)


def _string_list(value: Any, field_name: str, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    _require(isinstance(value, list), f"{field_name} must be a list of strings")
    for item in value:
        _require(isinstance(item, str) and item.strip(), f"{field_name} entries must be non-empty strings")
    return tuple(item.strip() for item in value)


def load_evaluation_set(path: str | Path = DEFAULT_EVAL_PATH) -> EvaluationSet:
    """Load and structurally validate the dataset at ``path``."""
    dataset_path = Path(path).expanduser()
    try:
        raw = dataset_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationSetError(f"cannot read evaluation set {dataset_path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvaluationSetError(f"evaluation set {dataset_path} is not valid JSON: {exc}") from exc
    _require(isinstance(data, dict), "the evaluation set must be a JSON object")

    name = data.get("name")
    version = data.get("version")
    _require(isinstance(name, str) and name.strip(), "evaluation set needs a non-empty 'name'")
    _require(isinstance(version, str) and version.strip(), "evaluation set needs a non-empty 'version'")
    cases_raw = data.get("cases")
    _require(isinstance(cases_raw, list) and cases_raw, "evaluation set needs a non-empty 'cases' list")

    cases: list[EvalCase] = []
    for index, item in enumerate(cases_raw):
        _require(isinstance(item, dict), f"case #{index} must be an object")
        case_id = item.get("id")
        topic = item.get("topic")
        _require(isinstance(case_id, str) and case_id.strip(), f"case #{index} needs a non-empty 'id'")
        _require(isinstance(topic, str) and topic.strip(), f"case {case_id!r} needs a non-empty 'topic'")
        kind = item.get("kind", "research")
        expect = item.get("expect", "accept")
        _require(kind in _CASE_KINDS, f"case {case_id!r} has unknown kind {kind!r}")
        _require(expect in _CASE_EXPECTATIONS, f"case {case_id!r} has unknown expect {expect!r}")
        min_grounding = item.get("min_grounding")
        if min_grounding is not None:
            _require(
                isinstance(min_grounding, (int, float)) and 0.0 <= float(min_grounding) <= 1.0,
                f"case {case_id!r} min_grounding must be between 0 and 1",
            )
        cases.append(
            EvalCase(
                id=case_id.strip(),
                topic=topic.strip(),
                kind=kind,
                expect=expect,
                required_tracks=_string_list(item.get("required_tracks"), f"case {case_id!r} required_tracks"),
                min_sections=int(item.get("min_sections", 1)),
                min_sources=int(item.get("min_sources", 1)),
                min_findings=int(item.get("min_findings", 1)),
                min_grounding=None if min_grounding is None else float(min_grounding),
                require_citations=bool(item.get("require_citations", True)),
                forbid_terms=_string_list(item.get("forbid_terms"), f"case {case_id!r} forbid_terms"),
                tags=_string_list(item.get("tags"), f"case {case_id!r} tags"),
                notes=str(item.get("notes", "")),
            )
        )

    dataset = EvaluationSet(
        name=name.strip(),
        version=version.strip(),
        cases=tuple(cases),
        created=str(data.get("created", "")),
        description=str(data.get("description", "")),
        path=str(dataset_path),
    )
    errors = validate_evaluation_set(dataset)
    if errors:
        raise EvaluationSetError("; ".join(errors))
    return dataset


def validate_evaluation_set(dataset: EvaluationSet) -> list[str]:
    """Return every structural or domain-lock inconsistency in ``dataset``."""
    errors: list[str] = []
    seen: set[str] = set()
    for case in dataset.cases:
        if case.id in seen:
            errors.append(f"duplicate case id {case.id!r}")
        seen.add(case.id)
        if case.expect == "accept":
            try:
                ensure_domain_topic(case.topic)
            except DomainError as exc:
                errors.append(f"case {case.id!r} expects acceptance but is off-domain: {exc}")
        else:
            try:
                ensure_domain_topic(case.topic)
            except DomainError:
                pass
            else:
                errors.append(
                    f"case {case.id!r} expects rejection but {case.topic!r} is in-domain"
                )
        unknown = sorted(set(case.required_tracks) - set(DOMAIN_TRACK_KEYS))
        if unknown:
            errors.append(
                f"case {case.id!r} names non-canonical track(s): {', '.join(unknown)}"
            )
        if case.min_sections < 0 or case.min_sources < 0 or case.min_findings < 0:
            errors.append(f"case {case.id!r} has a negative minimum")
        if case.kind == "injection" and not case.forbid_terms:
            errors.append(
                f"case {case.id!r} is an injection probe but lists no forbid_terms"
            )
    return errors


def _report_payload(report: Any) -> dict[str, Any]:
    if isinstance(report, dict):
        return report
    dump = getattr(report, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    raise TypeError("report must be a mapping or a pydantic model")


def _section_texts(report: dict[str, Any]) -> list[str]:
    sections = report.get("sections") or []
    texts: list[str] = []
    for section in sections:
        if isinstance(section, dict):
            texts.append(str(section.get("section_markdown") or ""))
        else:
            texts.append(str(getattr(section, "section_markdown", "") or ""))
    return texts


def _finding_count(report: dict[str, Any]) -> int:
    total = 0
    for section in report.get("sections") or []:
        findings = (
            section.get("findings")
            if isinstance(section, dict)
            else getattr(section, "findings", [])
        )
        total += len(findings or [])
    return total


def _track_keys(report: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for section in report.get("sections") or []:
        track = section.get("track") if isinstance(section, dict) else getattr(section, "track", None)
        if track:
            keys.add(str(track))
    return keys


def _audit_ratio(report: dict[str, Any], name: str) -> float | None:
    audit = report.get("audit") or {}
    value = audit.get(name) if isinstance(audit, dict) else getattr(audit, name, None)
    if value is None:
        metrics = report.get("metrics") or {}
        value = metrics.get(name)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _all_text(report: dict[str, Any]) -> str:
    chunks = [
        str(report.get("title") or ""),
        str(report.get("executive_summary") or ""),
        str(report.get("draft_markdown") or ""),
        str(report.get("methodology") or ""),
        *(_section_texts(report)),
    ]
    for section in report.get("sections") or []:
        findings = (
            section.get("findings")
            if isinstance(section, dict)
            else getattr(section, "findings", [])
        )
        for finding in findings or []:
            text = finding.get("text") if isinstance(finding, dict) else getattr(finding, "text", "")
            chunks.append(str(text or ""))
    return "\n".join(chunks)


def evaluate_case(report: Any | None, case: EvalCase) -> CaseResult:
    """Evaluate one saved report against one research/injection case."""
    if not case.needs_report:
        # Gate cases are evaluated by :func:`evaluate_dataset`.
        return CaseResult(
            case_id=case.id,
            topic=case.topic,
            kind=case.kind,
            status="skipped",
            message="gate case (no report required)",
        )
    if report is None:
        return CaseResult(
            case_id=case.id,
            topic=case.topic,
            kind=case.kind,
            status="skipped",
            message="no saved report matched this case",
        )

    payload = _report_payload(report)
    checks: list[EvalCheck] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append(EvalCheck(name=name, ok=bool(ok), detail=detail))

    check(
        "domain",
        str(payload.get("domain") or "") == DOMAIN,
        f"domain={payload.get('domain')!r}",
    )
    check(
        "topic",
        _normalise_topic(str(payload.get("topic") or "")) == _normalise_topic(case.topic),
        f"topic={payload.get('topic')!r}",
    )
    sections = _section_texts(payload)
    check(
        "sections",
        len(sections) >= case.min_sections,
        f"{len(sections)} section(s), need {case.min_sections}",
    )
    missing = sorted(set(case.required_tracks) - _track_keys(payload))
    check("required_tracks", not missing, f"missing: {', '.join(missing)}" if missing else "all present")
    sources = payload.get("sources") or []
    check(
        "sources",
        len(sources) >= case.min_sources,
        f"{len(sources)} source(s), need {case.min_sources}",
    )
    findings = _finding_count(payload)
    check(
        "findings",
        findings >= case.min_findings,
        f"{findings} finding(s), need {case.min_findings}",
    )

    if case.require_citations:
        known_ids = {
            str(source.get("id") if isinstance(source, dict) else getattr(source, "id", ""))
            for source in sources
        }
        cited: set[str] = set()
        uncited_sections: list[int] = []
        dangling: set[str] = set()
        for index, text in enumerate(sections):
            ids = {"S" + match for match in _CITATION_RE.findall(text)}
            if not ids:
                uncited_sections.append(index)
            cited |= ids
            dangling |= ids - known_ids
        check(
            "citations",
            not uncited_sections and not dangling,
            (
                f"uncited sections {uncited_sections}; dangling {sorted(dangling)}"
                if uncited_sections or dangling
                else f"{len(cited)} distinct citation id(s)"
            ),
        )

    if case.min_grounding is not None:
        grounding = _audit_ratio(payload, "grounded_ratio")
        check(
            "grounding",
            grounding is not None and grounding >= case.min_grounding,
            f"grounding={grounding} need>={case.min_grounding}",
        )

    text = _all_text(payload)
    # The case topic is itself the injection payload in an injection probe, so
    # the title/executive summary legitimately echo it.  Strip the exact topic
    # before scanning for leaked instructions or credentials; anything the model
    # adds beyond the topic is still caught.
    if case.kind == "injection" and case.topic:
        text = re.sub(re.escape(case.topic), " ", text, flags=re.IGNORECASE)
    lowered = text.lower()
    forbidden = [term for term in case.forbid_terms if term.lower() in lowered]
    check(
        "forbidden_terms",
        not forbidden,
        f"found: {', '.join(forbidden)}" if forbidden else "none",
    )
    leaked = [marker for marker in _SECRET_MARKERS if marker in lowered]
    check(
        "no_secret_leak",
        not leaked,
        f"found: {', '.join(leaked)}" if leaked else "none",
    )

    failed = [item.name for item in checks if not item.ok]
    return CaseResult(
        case_id=case.id,
        topic=case.topic,
        kind=case.kind,
        status="fail" if failed else "pass",
        checks=checks,
        message="failed: " + ", ".join(failed) if failed else "all checks passed",
    )


def _gate_case(case: EvalCase) -> CaseResult:
    checks: list[EvalCheck] = []
    try:
        normalised = ensure_domain_topic(case.topic)
        accepted = True
        detail = normalised
    except DomainError as exc:
        accepted = False
        detail = str(exc)
    expected = case.expect == "accept"
    checks.append(
        EvalCheck(
            name="domain_gate",
            ok=accepted is expected,
            detail=f"accepted={accepted} expected={expected}: {detail}",
        )
    )
    return CaseResult(
        case_id=case.id,
        topic=case.topic,
        kind=case.kind,
        status="pass" if all(check.ok for check in checks) else "fail",
        checks=checks,
        message=checks[0].detail,
    )


def evaluate_dataset(
    dataset: EvaluationSet,
    reports: Iterable[Any] = (),
) -> EvaluationSummary:
    """Evaluate every case, matching saved reports to research cases by topic."""
    by_topic: dict[str, Any] = {}
    for report in reports:
        payload = _report_payload(report)
        key = _normalise_topic(str(payload.get("topic") or ""))
        if key:
            by_topic[key] = report

    summary = EvaluationSummary(dataset=dataset.name, version=dataset.version)
    for case in dataset.cases:
        if case.needs_report:
            report = by_topic.get(_normalise_topic(case.topic))
            summary.results.append(evaluate_case(report, case))
        else:
            summary.results.append(_gate_case(case))
    return summary


def _metric(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    try:
        return None if current is None else float(current)
    except (TypeError, ValueError):
        return None


def _first_metric(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Return the first present metric from ``keys`` (new vocabulary first)."""
    for key in keys:
        value = _metric(payload, (key,))
        if value is not None:
            return value
    return None


def compare_metrics(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    max_drop: float = 0.05,
    max_cost_increase: float = 0.25,
    max_token_increase: float = 0.35,
) -> list[str]:
    """Return human-readable regressions between a baseline and a new run.

    ``baseline`` and ``current`` may be report metrics or full report payloads
    (the function looks inside ``metrics`` when present).  Grounding and
    entailment may not fall by more than ``max_drop``; estimated cost and token
    use may not rise by more than the configured fractions.  An unknown metric
    is skipped rather than treated as a regression, because a free-tier model
    reports zero cost.
    """
    baseline_metrics = baseline.get("metrics") if "metrics" in baseline else baseline
    current_metrics = current.get("metrics") if "metrics" in current else current
    if not isinstance(baseline_metrics, dict) or not isinstance(current_metrics, dict):
        return ["cannot compare: metrics are not mappings"]

    regressions: list[str] = []
    for label, path in (
        ("grounding", ("audit", "grounded_ratio")),
        ("entailment", ("audit", "entailment_ratio")),
    ):
        before = _metric(baseline, path)
        after = _metric(current, path)
        if before is None or after is None:
            # Report metrics store the audit ratio at the top level too.
            before = before if before is not None else _metric(baseline_metrics, (path[-1],))
            after = after if after is not None else _metric(current_metrics, (path[-1],))
        if before is None or after is None:
            continue
        if after < before - max_drop:
            regressions.append(
                f"{label} regressed: {before:.3f} -> {after:.3f} "
                f"(allowed drop {max_drop:.3f})"
            )

    for label, key, allowed in (
        ("cost", "estimated_cost_usd", max_cost_increase),
        ("tokens", "total_tokens", max_token_increase),
    ):
        before = _metric(baseline_metrics, (key,))
        after = _metric(current_metrics, (key,))
        if before is None or after is None or before <= 0:
            continue
        if after > before * (1.0 + allowed):
            regressions.append(
                f"{label} increased: {before:.4g} -> {after:.4g} "
                f"(allowed +{allowed:.0%})"
            )

    before_findings = _first_metric(baseline_metrics, ("findings_recorded", "claims"))
    after_findings = _first_metric(current_metrics, ("findings_recorded", "claims"))
    if before_findings and after_findings is not None and after_findings < before_findings:
        regressions.append(
            f"findings decreased: {before_findings:.0f} -> {after_findings:.0f}"
        )
    return regressions
