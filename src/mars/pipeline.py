"""The research pipeline: plan -> research -> analyse -> write -> edit -> audit.

The pipeline is deliberately explicit about control flow.  Each stage is a
CrewAI crew, the pipeline decides what runs when, and every intermediate
artefact is kept so a run can be inspected (and re-run) after the fact.

A typical run looks like this::

    plan ──▶ ┌─ track 1 ─┐
             ├─ track 2 ─┼─▶ analyse ─▶ write ─┐
             └─ track N ─┘                     │
                                               ▼
                                    edit ─▶ audit ─▶ (gaps?) ─▶ edit ...
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agents import (
    dossier_problem,
    run_analysis,
    run_audit,
    run_director,
    run_editing,
    run_gap_research,
    run_research,
    run_writing,
    section_citation_problem,
)
from .agents.contracts import contract_manifest, validate_contracts
from .agents.prompt_registry import active_prompts, configure_prompts_from_settings
from .agents.tasks import track_title
from .approval import ApprovalGate
from .config import Settings
from .context import ResearchContext
from .data_governance import disclosure as _disclosure
from .domain import (
    DEFAULT_DOMAIN_TRACKS,
    DOMAIN,
    DOMAIN_TRACK_KEYS,
    DomainError,
    canonical_track,
    ensure_domain_topic,
)
from .evidence import EvidenceStore
from .llm import build_llm
from .metrics import Stopwatch, UsageTracker, usage_scope
from .model_registry import ModelRegistryError, model_manifest, resolve_model_chain
from .observability import send_alert
from .reliability import ProviderUnavailableError
from .reporting import render_report
from .schemas import (
    AuditResult,
    Claim,
    EvaluationResult,
    FindingBundle,
    ResearchPlan,
    ResearchReport,
    Source,
    TimelineEntry,
    TrackPlan,
    TrackResult,
)
from .storage import atomic_write_json, atomic_write_text, read_json, unique_run_id

__all__ = ["ResearchPipeline", "PipelineResult"]

Reporter = Callable[[str], None]

# Presentation order of the canonical Agentic AI taxonomy.
DOMAIN_TRACK_ORDER = {key: index for index, key in enumerate(DOMAIN_TRACK_KEYS)}


class ResearchTrackIncomplete(RuntimeError):
    """A track ended without evidence a section could be written from.

    Distinct from a provider outage: the call succeeded, but the result was a
    tool call in prose, an uncited dossier, or a section with no citations.
    The track is recorded as not covered instead of being padded with prose.
    """


def coverage_requirement(planned: int) -> int:
    """Minimum number of evidence-bearing tracks for a usable report.

    A report built from one surviving track is not a research result: it
    silently drops most of the field.  Half the plan (and never fewer than
    two tracks, except for a one-track plan) is the floor.
    """
    if planned <= 1:
        return 1
    return max(2, math.ceil(planned * 0.5))


@dataclass(slots=True)
class PipelineResult:
    """Everything a run produced."""

    report: ResearchReport
    report_markdown: str
    plan: ResearchPlan
    evidence: EvidenceStore
    usage: UsageTracker
    output_paths: dict[str, str] = field(default_factory=dict)
    elapsed: float = 0.0


class ResearchPipeline:
    """Runs a full multi-agent research programme."""

    def __init__(
        self,
        settings: Settings,
        *,
        context: ResearchContext | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        self.settings = settings
        self.reporter: Reporter = reporter or (lambda message: None)
        self.usage = UsageTracker(
            max_tokens=settings.max_tokens_per_run,
            max_cost_usd=settings.max_cost_usd,
            max_tokens_per_track=settings.max_tokens_per_track,
            max_cost_usd_per_track=settings.max_cost_usd_per_track,
        )
        self._lock = threading.Lock()
        self.run_id = settings.run_id or unique_run_id()
        # Give the context (and therefore its durable DLQ) the same stable run
        # identifier used by persisted artefacts.
        self.context = context or ResearchContext(settings=settings.with_overrides(run_id=self.run_id))
        self.approval = ApprovalGate(settings, reporter=self.reporter)
        self._allocations: dict[str, int] = {}
        # Set when the editor agent was unavailable and the draft had to be
        # assembled mechanically from the track sections.
        self._edit_degraded = False

    # ------------------------------------------------------------------ utils
    def _log(self, message: str) -> None:
        self.reporter(message)

    def _trace(self, name: str, *, status: str = "ok", **attributes: object) -> None:
        if self.context.observer is not None:
            self.context.observer.emit(name, status=status, **attributes)

    def _alert(self, kind: str, message: str, **fields: object) -> None:
        send_alert(self.settings.alert_webhook_url, kind, message, run_id=self.run_id, **fields)

    def _llm(self, role: str):
        return build_llm(self.settings, role, usage=self.usage, context=self.context)

    def _checkpoint(self, stage: str, status: str, summary: dict[str, object] | None = None) -> None:
        if self.context.run_state is not None:
            self.context.run_state.transition(stage, status, summary=summary)
        self._trace("stage.transition", stage=stage, stage_status=status, summary=summary or {})

    def _guard(self, stage: str) -> None:
        if self.context.watchdog is None:
            return
        try:
            self.context.watchdog.check_run()
        except Exception as exc:
            self._checkpoint(stage, "timed_out", {"error": str(exc)})
            self._dead_letter(stage, exc, {"topic": self.settings.topic})
            raise

    # -------------------------------------------------------------------- run
    def request_shutdown(self) -> None:
        """Ask every cooperating stage to stop at its next safe boundary."""
        if self.context.watchdog is not None:
            self.context.watchdog.request_stop()
        self._trace("run.shutdown_requested")

    def run(self, *, resume: bool = False) -> PipelineResult:
        """Execute the programme, recording a terminal state on failure."""
        try:
            return self._run(resume=resume)
        except BaseException as exc:
            if self.context.run_state is not None:
                self.context.run_state.finish(
                    "failed", summary={"error_type": type(exc).__name__, "error": str(exc)[:2000]}
                )
            self._trace("run.failed", status="error", error_type=type(exc).__name__)
            self._alert(
                "run_failed",
                f"run {self.run_id} failed: {type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
            )
            raise

    def _run(self, *, resume: bool = False) -> PipelineResult:
        started = time.perf_counter()
        settings = self.settings
        settings.validate()
        settings = self._resolve_models(settings)
        configure_prompts_from_settings(settings)
        contract_errors = validate_contracts()
        if contract_errors:
            raise RuntimeError(
                "agent contracts are inconsistent: " + "; ".join(contract_errors)
            )
        state = self.context.run_state
        if resume:
            if state is None or not state.path.is_file():
                raise RuntimeError(f"no run state found for run_id {self.run_id!r}")
            if state.status == "completed":
                raise RuntimeError(f"run {self.run_id!r} already completed")
            persisted_topic = (state.stage_summary("run") or {}).get("topic")
            if isinstance(persisted_topic, str) and persisted_topic:
                if persisted_topic != settings.topic:
                    self._log(
                        f"Resuming with the persisted topic {persisted_topic!r} "
                        f"(requested {settings.topic!r})"
                    )
                # A persisted run from before the domain lock (or a tampered
                # state file) must not be able to resume off-domain.
                try:
                    persisted_topic = ensure_domain_topic(persisted_topic)
                except DomainError as exc:
                    raise RuntimeError(str(exc)) from exc
                settings = settings.with_overrides(topic=persisted_topic)
                self.settings = settings
                self.context.settings = settings

        # Fail closed: a missing key must never silently turn a live deployment
        # into a degraded one that emits plausible-looking but fabricated research.
        if settings.llm_provider != "openrouter" or not settings.openrouter_api_key:
            raise RuntimeError(
                "the openrouter provider with OPENROUTER_API_KEY is required; "
                "there is no mock/offline fallback"
            )

        self._log(f"Topic: {settings.topic}")
        self._log(
            f"Mode: live LLM via OpenRouter ({settings.model_strong})"
            + ("" if settings.allow_network else " - network tools disabled")
        )

        if resume:
            self._restore_sources(settings.topic)
        self._checkpoint("run", "started", {"topic": settings.topic})
        self._trace("run.started", topic=settings.topic, depth=settings.depth)

        # -- Stage 0: plan -------------------------------------------------
        resumed_plan: ResearchPlan | None = None
        if resume and state is not None:
            summary = state.stage_summary("plan") or {}
            persisted = summary.get("plan")
            if isinstance(persisted, dict):
                try:
                    resumed_plan = ResearchPlan.model_validate(persisted)
                except Exception:
                    resumed_plan = None
            if resumed_plan is None:
                persisted = read_json(self._plan_path(settings.topic))
                if isinstance(persisted, dict):
                    try:
                        resumed_plan = ResearchPlan.model_validate(persisted)
                    except Exception:
                        resumed_plan = None

        if resumed_plan is not None:
            plan = self._normalise_plan(resumed_plan)
            plan_timer = Stopwatch("plan")
            self._log(f"Resuming with the persisted plan ({len(plan.tracks)} tracks)")
        else:
            with Stopwatch("plan") as plan_timer:
                self._checkpoint("plan", "running")
                plan = self._plan()
            plan = self._normalise_plan(plan)
        self._persist_plan(plan)
        self._checkpoint(
            "plan",
            "completed",
            {"tracks": [track.track for track in plan.tracks], "plan": plan.model_dump(mode="json")},
        )
        # Store plan for _ordered_results to use track ordering
        self._last_plan = plan
        if resumed_plan is None:
            self._log(f"Plan ready with {len(plan.tracks)} tracks ({plan_timer.elapsed:.1f}s)")
        self.approval.request_approval(
            "plan",
            self.run_id,
            settings.topic,
            {
                "tracks": [t.track for t in plan.tracks],
                "framing": plan.framing,
                "track_details": [
                    {"track": t.track, "objective": t.objective, "questions": t.key_questions}
                    for t in plan.tracks
                ],
            },
            payload=plan.model_dump(mode="json"),
        )

        # -- Stage 1: research the tracks in parallel ------------------------
        resume_results = self._restore_track_results(plan, state) if resume else {}
        track_results = self._research_tracks(plan, resume_results=resume_results)
        # Coverage is measured by evidence, not by "the track did not raise".
        # A track whose dossier was a tool call in prose has no findings and no
        # ledger claims; counting it as complete is how four empty sections
        # reached a finished report.
        covered_tracks = [
            result
            for result in track_results.values()
            if not result.warnings
            and (
                result.findings
                or self.context.evidence.claims(track=result.track)
            )
        ]
        covered_names = {result.track for result in covered_tracks}
        failed_tracks = [
            track for track in track_results if track not in covered_names
        ]
        planned_tracks = len(plan.tracks) or len(self.settings.tracks)
        required_tracks = coverage_requirement(planned_tracks)
        if len(covered_tracks) < required_tracks:
            if not covered_tracks:
                # Nothing was researched. Producing a report made of failure
                # stubs is worse than a clear stop: say what happened and where
                # to look.
                detail = next(
                    (
                        result.warnings[0]
                        for result in track_results.values()
                        if result.warnings
                    ),
                    "no track produced evidence",
                )
                raise ProviderUnavailableError(
                    "every research track failed before producing evidence; the model "
                    f"provider is unavailable or rate-limited ({detail}). See the "
                    "dead-letter file for the provider error. If this is an OpenRouter "
                    "free-tier quota, wait for the window to reset or configure another "
                    "model with MARS_MODEL_STRONG / MARS_MODEL_FAST / MARS_MODEL_PLANNER."
                )
            raise ProviderUnavailableError(
                f"only {len(covered_tracks)} of {planned_tracks} tracks produced "
                f"evidence (minimum {required_tracks}). The remaining tracks "
                f"({', '.join(sorted(failed_tracks))}) failed before recording a "
                "cited finding, so a report would silently drop that part of the "
                "field. See the dead-letter file for the provider error and retry "
                "with --resume once it is resolved."
            )
        if failed_tracks:
            self._request_degraded_approval(
                "track failures",
                {"failed_tracks": failed_tracks, "covered": len(covered_tracks)},
            )

        # -- Stage 2: assemble the report -----------------------------------
        draft = self._edit(plan, track_results)
        self.approval.request_approval(
            "draft",
            self.run_id,
            settings.topic,
            {
                "draft_word_count": len(draft.split()),
                "sources": len(self.context.evidence.sources()),
                "findings": sum(len(r.findings) for r in track_results.values()),
            },
            payload=draft,
        )
        audit = self._audit(draft)
        if audit.verdict.lower() in {"fail", "unreviewed"}:
            self._request_degraded_approval(
                "audit unavailable or failed", {"verdict": audit.verdict, "notes": audit.notes}
            )
        elif audit.degraded:
            self._request_degraded_approval(
                "audit did not verify the report",
                {"verdict": audit.verdict, "reasons": audit.degraded_reasons},
            )

        # -- Stage 3: close the gaps the auditor found ----------------------
        rounds = 1
        gap_log: list[str] = []
        while (
            rounds < settings.max_rounds
            and audit.gaps
            and audit.verdict.lower() in {"revise", "fail", "unreviewed"}
        ):
            self._log(
                f"Checking {len(audit.gaps)} gap(s) flagged by the auditor"
            )
            gaps = audit.gaps[:8]
            before_claims = len(self.context.evidence.claims())
            self._research_gaps(gaps)
            added_claims = len(self.context.evidence.claims()) - before_claims
            if added_claims < 1:
                # A gap round that records no new evidence cannot improve the
                # report. Re-editing and re-auditing would spend budget to
                # reproduce the same verdict, so stop at one round.
                self._log(
                    "Gap research added no ledger claims; keeping the report at "
                    f"{rounds} round(s) instead of re-editing it"
                )
                break
            rounds += 1
            self._log(
                f"Round {rounds}: gap research added {added_claims} ledger claim(s)"
            )
            gap_log.extend(gaps)
            # Give the editor the new evidence, then re-audit.
            draft = self._edit(plan, track_results, extra_notes=self._gap_notes(gap_log))
            self.approval.request_approval(
                "revision",
                self.run_id,
                settings.topic,
                {"round": rounds, "gaps_closed": len(gaps), "draft_word_count": len(draft.split())},
                payload=draft,
            )
            audit = self._audit(draft)
            if audit.degraded:
                self._request_degraded_approval(
                    "revised report did not verify",
                    {"round": rounds, "reasons": audit.degraded_reasons},
                )

        report = self._build_report(plan, track_results, draft, audit, rounds)
        evaluation = self._evaluate(report)
        if evaluation is not None:
            report.metrics["continuous_evaluation"] = evaluation.model_dump(mode="json")
        elapsed = time.perf_counter() - started
        report.metrics["wall_clock_seconds"] = round(elapsed, 2)
        report.metrics["rounds"] = rounds
        markdown = render_report(report, self.context.evidence)

        paths = self._persist(report, markdown, plan)
        if self.context.run_state is not None:
            self.context.run_state.finish(
                "completed",
                summary={
                    "grounding": audit.grounded_ratio,
                    "entailment": audit.entailment_ratio,
                    "audit_status": audit.status,
                    "failed_tracks": failed_tracks,
                },
            )
        self._trace(
            "run.completed",
            sources=len(self.context.evidence.sources()),
            total_tokens=self.usage.total_tokens,
            estimated_cost_usd=round(self.usage.cost, 6),
            circuit=self.context.llm_breaker.snapshot() if self.context.llm_breaker else {},
        )
        self._log(
            f"Done in {elapsed:.1f}s - "
            f"{len(self.context.evidence.sources())} sources, "
            f"{self.usage.total_tokens} tokens, "
            f"grounding {audit.grounded_ratio:.0%}, "
            f"entailment {audit.entailment_ratio:.0%} ({audit.status})"
        )

        return PipelineResult(
            report=report,
            report_markdown=markdown,
            plan=plan,
            evidence=self.context.evidence,
            usage=self.usage,
            output_paths=paths,
            elapsed=elapsed,
        )

    # --------------------------------------------------------------- stage 0
    def _resolve_models(self, settings: Settings) -> Settings:
        """Substitute an active fallback for any retired configured model.

        Validation has already guaranteed that a retired primary has a usable
        fallback.  Resolving here means the run manifest records the model that
        actually ran, not the retired id that was configured.
        """
        updates: dict[str, str] = {}
        for role in ("strong", "fast", "planner"):
            primary = getattr(settings, f"model_{role}")
            try:
                chain = resolve_model_chain(primary, tuple(settings.model_fallbacks))
            except ModelRegistryError:  # pragma: no cover - validate() guards this
                continue
            if chain[0] != primary:
                updates[f"model_{role}"] = chain[0]
                self._log(
                    f"Model {primary!r} is retired; using fallback {chain[0]!r} for {role}"
                )
        if not updates:
            return settings
        resolved = settings.with_overrides(**updates)
        self.settings = resolved
        self.context.settings = resolved
        return resolved

    def _plan(self) -> ResearchPlan:
        self._guard("plan")
        llm = self._llm("planner")
        try:
            # Pass None to let the Director determine optimal tracks for the topic
            with usage_scope(agent="director", track="plan"):
                return run_director(llm, self.settings.topic, None)
        except ProviderUnavailableError as exc:
            # A spent quota or a dead provider will fail every later call too.
            # Stop now instead of burning the remaining tracks on it.
            self._log(f"Planning failed ({exc})")
            self._dead_letter("plan", exc, {"topic": self.settings.topic})
            raise
        except Exception as exc:  # pragma: no cover - defensive
            self._log(f"Planning failed ({exc}); falling back to a default plan")
            self._dead_letter("plan", exc, {"topic": self.settings.topic})
            from .agents.factory import _recover_plan

            return _recover_plan("", self.settings.topic, list(self.settings.tracks))

    def _normalise_plan(self, plan: ResearchPlan) -> ResearchPlan:
        """Bound a model-generated fan-out to the fixed Agentic AI taxonomy.

        Anything the Director invents that cannot be mapped onto a canonical
        track is dropped rather than researched, so the report can never widen
        beyond the domain.
        """
        try:
            plan.topic = ensure_domain_topic(plan.topic or self.settings.topic)
        except DomainError:
            # A model-supplied topic cannot widen the run; keep the validated one.
            plan.topic = self.settings.topic
        plan.domain = DOMAIN
        unique: dict[str, TrackPlan] = {}
        for track in plan.tracks:
            canonical = canonical_track(track.track)
            if canonical is None:
                self._log(
                    f"Dropping off-taxonomy track {track.track!r}; "
                    f"Mars only researches {DOMAIN}"
                )
                continue
            track.track = canonical.key
            if not track.title:
                track.title = canonical.title
            if canonical.key not in unique:
                unique[canonical.key] = track
        plan.tracks = sorted(
            unique.values(), key=lambda item: DOMAIN_TRACK_ORDER.get(item.track, 99)
        )[: max(1, self.settings.max_tracks)]
        if not plan.tracks:
            from .agents.factory import _recover_plan

            return _recover_plan("", self.settings.topic, list(DEFAULT_DOMAIN_TRACKS))
        return plan

    # ------------------------------------------------------------- resumable
    def _artifact_base(self, topic: str) -> Path:
        return Path(self.settings.output_dir) / f"{_slug(topic)}-{self.run_id}"

    def _plan_path(self, topic: str) -> Path:
        return self._artifact_base(topic).with_name(
            f"{self._artifact_base(topic).name}-plan.json"
        )

    def _sources_path(self, topic: str) -> Path:
        return self._artifact_base(topic).with_name(
            f"{self._artifact_base(topic).name}-sources.json"
        )

    def _persist_plan(self, plan: ResearchPlan) -> None:
        atomic_write_json(self._plan_path(self.settings.topic), plan.model_dump(mode="json"))

    def _persist_sources(self, topic: str | None = None) -> None:
        atomic_write_json(
            self._sources_path(topic or self.settings.topic),
            [source.model_dump(mode="json") for source in self.context.evidence.sources()],
        )

    def _restore_sources(self, topic: str) -> bool:
        data = read_json(self._sources_path(topic))
        if not isinstance(data, list) or not data:
            return False
        try:
            sources = [Source.model_validate(item) for item in data]
        except Exception:
            return False
        self.context.evidence.restore(sources)
        self._log(f"Restored {len(sources)} sources from the previous attempt")
        return True

    def _restore_track_results(
        self, plan: ResearchPlan, state: object
    ) -> dict[str, TrackResult]:
        restored: dict[str, TrackResult] = {}
        if state is None:
            return restored
        for track_plan in plan.tracks:
            track = track_plan.track
            stage = f"track:{track}"
            if state.stage_status(stage) != "completed":
                continue
            summary = state.stage_summary(stage) or {}
            payload = summary.get("result")
            if not isinstance(payload, dict):
                continue
            try:
                restored[track] = TrackResult.model_validate(payload)
            except Exception:
                continue
        if restored:
            self._log(f"Reusing {len(restored)} completed track(s) from the previous attempt")
        return restored

    # --------------------------------------------------------------- stage 1
    def _research_tracks(
        self,
        plan: ResearchPlan,
        *,
        resume_results: dict[str, TrackResult] | None = None,
    ) -> dict[str, TrackResult]:
        """Research every track from the plan, in parallel, and write its section.

        On a resumed run, tracks already marked ``completed`` in the persisted
        state journal are reused verbatim instead of being researched again.
        """
        plans = {track.track: track for track in plan.tracks}
        results: dict[str, TrackResult] = dict(resume_results or {})
        # Use tracks from the plan (Director-determined), fallback to settings
        tracks = [t.track for t in plan.tracks] or list(self.settings.tracks)
        workers = max(1, min(self.settings.max_parallel_tracks, len(tracks)))
        # Allocation is stable across completion order and scheduler timing.
        self._allocations = {track: index % workers for index, track in enumerate(tracks)}
        pending = [track for track in tracks if track not in results]
        if len(pending) < len(tracks):
            self._log(f"Skipping {len(tracks) - len(pending)} already-completed track(s)")
        if not pending:
            return results

        def research_one(track: str) -> tuple[str, TrackResult]:
            started = time.monotonic()
            self._checkpoint(f"track:{track}", "running", {"worker": self._allocations[track]})
            track_plan = plans.get(track)
            plan_dict = track_plan.model_dump() if track_plan else None
            llm = self._llm("fast")
            with usage_scope(agent="researcher", track=track):
                dossier = run_research(
                    self.context,
                    llm,
                    self.settings.topic,
                    track,
                    plan_dict,
                    verbose=self.settings.crew_verbose,
                )
            problem = dossier_problem(dossier)
            if problem:
                # A tool call in prose, or a dossier with no citations, means
                # the researcher never gathered evidence. Give it one explicit
                # repair attempt, then fail the track instead of letting the
                # writer invent a section.
                self._log(f"Track '{track}' dossier rejected ({problem}); retrying once")
                with usage_scope(agent="researcher", track=track):
                    dossier = run_research(
                        self.context,
                        llm,
                        self.settings.topic,
                        track,
                        plan_dict,
                        repair=True,
                        verbose=self.settings.crew_verbose,
                    )
                problem = dossier_problem(dossier)
            if problem:
                raise ResearchTrackIncomplete(f"{problem}; no section was written")
            if self.context.watchdog is not None:
                self.context.watchdog.check_track(started, track)
            with usage_scope(agent="analyst", track=track):
                bundle: FindingBundle = run_analysis(llm, track, dossier)
            if self.context.watchdog is not None:
                self.context.watchdog.check_track(started, track)

            findings = [finding for finding in bundle.findings if finding.evidence]
            ledger = self.context.evidence.claims(track=track)
            if not findings and ledger:
                # The claim ledger is the source of truth. An analyst that
                # returned nothing usable must not erase evidence the researcher
                # already recorded.
                findings = [
                    Claim(
                        text=claim.text,
                        evidence=list(claim.evidence),
                        confidence=claim.confidence,
                        kind=claim.kind,
                    )
                    for claim in ledger
                    if claim.evidence
                ]
                if findings:
                    self._log(
                        f"Track '{track}': analyst returned no usable findings; "
                        f"using {len(findings)} from the claim ledger"
                    )
            # A source flagged for prompt injection must never be cited. Drop it
            # from the evidence handed to the writer; a finding that depended on
            # it alone is discarded rather than laundered into the report.
            blocked = self.context.evidence.flagged_source_ids()
            if blocked:
                clean: list[Any] = []
                for finding in findings:
                    kept = [cid for cid in finding.evidence if cid not in blocked]
                    if kept:
                        finding.evidence = kept
                        clean.append(finding)
                dropped = len(findings) - len(clean)
                if dropped:
                    self._log(
                        f"Track '{track}': dropped {dropped} finding(s) whose only "
                        "evidence was flagged for prompt injection"
                    )
                findings = clean
            if not findings:
                raise ResearchTrackIncomplete(
                    "no evidence-backed finding was produced; no section was written"
                )
            findings_json = json.dumps(
                {
                    "track": track,
                    "summary": bundle.summary,
                    "findings": [finding.model_dump(mode="json") for finding in findings],
                    "timeline": [entry.model_dump(mode="json") for entry in bundle.timeline],
                    "open_questions": bundle.open_questions,
                },
                indent=2,
            )
            # P2: the writer must see the snippets for this track, not just the
            # finding text. The allowed set is exactly what it was handed, so a
            # citation can always be traced back to a snippet it read.
            writer_ids = {
                cid for finding in findings for cid in finding.evidence if cid not in blocked
            }
            writer_ids.update(
                cid
                for cid in self.context.evidence.extract_citations(dossier)
                if cid not in blocked
            )
            writer_sources = [
                source
                for cid in sorted(writer_ids, key=lambda item: int(item.lstrip("S") or 0))
                if (source := self.context.evidence.get(cid)) is not None
            ]
            sources_block = ""
            if writer_sources:
                from .agents.tasks import render_sources_block

                sources_block = (
                    f"SOURCES FOR THE {track.upper()} TRACK - "
                    f"{len(writer_sources)} source(s) with snippets:\n"
                    + render_sources_block(writer_sources, limit=None, snippet_chars=800)
                )
            allowed_ids = set(writer_ids)
            with usage_scope(agent="writer", track=track):
                section = run_writing(
                    llm,
                    self.settings.topic,
                    track,
                    dossier,
                    findings_json,
                    sources_block=sources_block,
                    forbidden_ids=sorted(blocked),
                    verbose=self.settings.crew_verbose,
                )
            section_problem = section_citation_problem(
                section, self.context.evidence, allowed_ids=allowed_ids
            )
            if section_problem:
                self._log(f"Track '{track}' section rejected ({section_problem}); retrying once")
                with usage_scope(agent="writer", track=track):
                    section = run_writing(
                        llm,
                        self.settings.topic,
                        track,
                        dossier,
                        findings_json,
                        repair=True,
                        sources_block=sources_block,
                        forbidden_ids=sorted(blocked),
                        verbose=self.settings.crew_verbose,
                    )
                section_problem = section_citation_problem(
                    section, self.context.evidence, allowed_ids=allowed_ids
                )
            if section_problem:
                raise ResearchTrackIncomplete(
                    f"{section_problem}; the section was not added to the report"
                )
            if self.context.watchdog is not None:
                self.context.watchdog.check_track(started, track)
            source_ids = sorted(
                {
                    cid
                    for finding in findings
                    for cid in finding.evidence
                }
            )
            result = TrackResult(
                track=track,
                title=track_title(track, self.settings.topic),
                dossier=dossier,
                section_markdown=section,
                findings=findings,
                source_ids=source_ids,
                open_questions=bundle.open_questions,
                timeline=bundle.timeline,
            )
            return track, result

        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mars-track")
        futures = {pool.submit(research_one, track): track for track in pending}
        try:
            remaining = max(0.0, self.settings.max_run_seconds - (time.monotonic() - self.context.watchdog.started)) if self.context.watchdog else None
            for future in as_completed(futures, timeout=remaining):
                track = futures[future]
                try:
                    _, result = future.result()
                    results[track] = result
                    self._checkpoint(
                        f"track:{track}",
                        "completed",
                        {
                            "findings": len(result.findings),
                            "result": result.model_dump(mode="json"),
                        },
                    )
                    # Persist sources after every completed track so a crash
                    # mid-fan-out can still resume with a stable evidence base.
                    self._persist_sources()
                    self._log(
                        f"Track '{track}' complete: "
                        f"{len(result.findings)} findings, "
                        f"{len(result.section_markdown.split())} words"
                    )
                except Exception as exc:
                    self._log(f"Track '{track}' failed: {exc}")
                    self._checkpoint(f"track:{track}", "failed", {"error": str(exc)})
                    self._dead_letter("research_track", exc, {"topic": self.settings.topic, "track": track})
                    # No stub section: a track without evidence is reported in
                    # the "Tracks not covered" table, never as prose.
                    results[track] = TrackResult(
                        track=track,
                        title=track_title(track, self.settings.topic),
                        warnings=[f"Track failed: {exc}"],
                        section_markdown="",
                    )
        except TimeoutError:
            for future, track in futures.items():
                if track not in results:
                    future.cancel()
                    error = TimeoutError(f"run watchdog expired before track '{track}' completed")
                    self._checkpoint(f"track:{track}", "timed_out", {"error": str(error)})
                    self._dead_letter("research_track", error, {"topic": self.settings.topic, "track": track})
                    results[track] = TrackResult(track=track, warnings=[f"Track timed out: {error}"])
        finally:
            # Do not hold report completion hostage to a non-cooperative SDK
            # call. Its request timeout and shared call budget still bound it.
            pool.shutdown(wait=False, cancel_futures=True)
        return results

    # --------------------------------------------------------------- stage 2
    def _edit(
        self,
        plan: ResearchPlan,
        track_results: dict[str, TrackResult],
        *,
        extra_notes: str | None = None,
    ) -> str:
        self._guard("edit")
        self._checkpoint("edit", "running")
        ordered = self._ordered_results(track_results)
        # Only evidence-backed sections reach the editor. A track that failed
        # its dossier or citation gate is listed as not covered instead.
        usable = [
            result
            for result in ordered
            if (result.section_markdown or "").strip() and not result.warnings
        ]
        skipped = [result for result in ordered if result not in usable]
        sections = "\n\n----------\n\n".join(
            result.section_markdown for result in usable
        )
        findings = [finding for result in usable for finding in result.findings]
        timeline = [entry for result in usable for entry in result.timeline]
        title = self._report_title()
        llm = self._llm("strong")
        notes = f"\n\nNEW EVIDENCE FROM THE FOLLOW-UP ROUND:\n{extra_notes}" if extra_notes else ""
        track_titles = [result.title or result.track.title() for result in usable]
        self._edit_degraded = False
        try:
            with usage_scope(agent="editor", track="report"):
                draft = run_editing(
                    llm,
                    self.settings.topic,
                    title,
                    sections + notes,
                    json.dumps([f.model_dump(mode="json") for f in findings], indent=2),
                    json.dumps([t.model_dump(mode="json") for t in timeline], indent=2),
                    track_titles=track_titles,
                    verbose=self.settings.crew_verbose,
                )
        except Exception as exc:
            # The research is already paid for. Losing the whole run because
            # the editorial pass failed (an outage, an open breaker, a spent
            # quota) is the wrong trade: assemble the sections we have and let
            # the audit mark the report degraded.
            self._edit_degraded = True
            self._log(
                f"Editing failed ({exc}); assembling the report from the "
                "completed track sections"
            )
            self._dead_letter("edit", exc, {"topic": self.settings.topic})
            self._checkpoint("edit", "degraded", {"error": str(exc)[:500]})
            return self._append_uncovered(
                self._assemble_draft(title, track_results), skipped
            )
        self._checkpoint("edit", "completed", {"words": len(draft.split())})
        return self._append_uncovered(draft, skipped)

    def _append_uncovered(self, draft: str, skipped: list[TrackResult]) -> str:
        """Name the tracks that produced no evidence, without writing for them.

        A Markdown table is deliberate: the grounding score ignores tables, so
        an honest coverage note does not count as uncited prose.
        """
        if not skipped:
            return draft
        rows = ["", "## Tracks not covered", "", "| Track | Reason |", "|---|---|"]
        for result in skipped:
            reason = (result.warnings[0] if result.warnings else "no evidence")
            reason = " ".join(reason.split()).replace("|", "/")[:200]
            rows.append(f"| {result.title or result.track.title()} | {reason} |")
        return draft.rstrip() + "\n" + "\n".join(rows) + "\n"

    def _assemble_draft(
        self, title: str, track_results: dict[str, TrackResult]
    ) -> str:
        """Build a citation-preserving draft without the editor agent."""
        ordered = self._ordered_results(track_results)
        usable = [
            result
            for result in ordered
            if (result.section_markdown or "").strip() and not result.warnings
        ]
        track_names = ", ".join(
            result.title or result.track.title() for result in usable
        ) or "none"
        parts = [
            f"# {title}",
            "",
            "## Executive Summary",
            "",
            (
                "This report was assembled without the editorial agent, which was "
                f"unavailable during the run. It contains {len(usable)} research "
                f"track(s): {track_names}. Each section below is the unedited output "
                "of its track and retains the original source citations."
            ),
            "",
            "> **Editorial note:** the editing stage failed, so the sections were "
            "concatenated mechanically. The audit below records the report as degraded.",
            "",
        ]
        for result in usable:
            parts.extend([result.section_markdown.strip(), ""])
        if not usable:
            parts.append(
                "No research track produced a section in this run. The failure is "
                "recorded in the run state and the dead-letter queue."
            )
        return "\n".join(parts).rstrip() + "\n"

    def _report_title(self) -> str:
        topic = self.settings.topic
        return f"{topic}: An Agentic AI Research Report"

    def _ordered_results(self, results: dict[str, TrackResult]) -> list[TrackResult]:
        # Order by the plan's track order if available, otherwise alphabetical
        plan_tracks = [t.track for t in getattr(self, '_last_plan', []).tracks] if hasattr(self, '_last_plan') and self._last_plan else []
        if plan_tracks:
            order = {track: index for index, track in enumerate(plan_tracks)}
        else:
            order = {}
        return sorted(
            results.values(), key=lambda result: order.get(result.track, 99)
        )

    # --------------------------------------------------------------- stage 3
    def _audit(self, draft: str) -> AuditResult:
        self._guard("audit")
        self._checkpoint("audit", "running")
        llm = self._llm("strong")
        from .agents.tasks import render_sources_block

        # The auditor can only judge the sources it is shown. Passing the first
        # 60 of 207 made every citation in the tail "unverifiable", which the
        # report then presented as weak evidence. Show exactly the sources the
        # draft cites, with enough snippet to judge support.
        all_sources = self.context.evidence.sources()
        cited_ids = self.context.evidence.extract_citations(draft)
        if cited_ids:
            wanted = set(cited_ids)
            selected = [source for source in all_sources if source.id in wanted]
            registry_note = (
                f"REGISTRY SCOPE: {len(selected)} of {len(all_sources)} registered "
                "sources - exactly those cited by the draft.\n"
            )
            sources_block = registry_note + render_sources_block(
                selected, limit=None, snippet_chars=800
            )
        else:
            sources_block = render_sources_block(all_sources, limit=60)
        try:
            with usage_scope(agent="auditor", track="report"):
                bundle = run_audit(llm, draft, sources_block, verbose=self.settings.crew_verbose)
        except Exception as exc:
            self._log(f"Audit failed ({exc}); falling back to mechanical checks only")
            self._dead_letter("audit", exc, {"topic": self.settings.topic})
            bundle = None

        mechanical_ratio, cited, total = self.context.evidence.grounding_score(draft)
        dangling = self.context.evidence.dangling_citations(draft)
        uncited = self.context.evidence.uncited_paragraphs(draft)
        entailment = self.context.evidence.entailment_audit(
            draft,
            supported_at=self.settings.entailment_supported_at,
            partial_at=self.settings.entailment_partial_at,
        )
        claim_report = self.context.evidence.verify_claims()

        degraded_reasons: list[str] = []
        if bundle is None:
            degraded_reasons.append("the auditor agent was unavailable")
            ratio = mechanical_ratio
            gaps = list(uncited)
            verdict = "unreviewed"
            notes = "Mechanical + entailment audit only (the auditor agent was unavailable)."
            cited_claims = cited
            total_claims = total
            unsupported = entailment.unsupported_claims
            method = "mechanical+entailment"
        else:
            ratio = min(float(bundle.grounded_ratio or 0.0), mechanical_ratio)
            gaps = [gap for gap in bundle.gaps if gap.strip()]
            verdict = bundle.verdict or "pass"
            notes = bundle.notes
            cited_claims = bundle.cited_claims or cited
            total_claims = bundle.total_claims or total
            unsupported = list(dict.fromkeys([*bundle.unsupported_claims, *entailment.unsupported_claims]))
            method = "llm+entailment"
            if dangling:
                verdict = "revise"
                degraded_reasons.append(f"{len(dangling)} dangling citation(s)")
            if ratio < self.settings.grounding_threshold:
                verdict = "revise"
                degraded_reasons.append(
                    f"grounding {ratio:.2f} below threshold {self.settings.grounding_threshold:.2f}"
                )

        if entailment.ratio < self.settings.min_entailment_ratio:
            verdict = "revise"
            degraded_reasons.append(
                f"entailment {entailment.ratio:.2f} below threshold "
                f"{self.settings.min_entailment_ratio:.2f}"
            )
        if entailment.unsupported:
            degraded_reasons.append(
                f"{entailment.unsupported} sentence(s) unsupported by their cited source"
            )
        if int(claim_report.get("unsupported", 0) or 0):
            degraded_reasons.append(
                f"{claim_report['unsupported']} ledger claim(s) unsupported by their cited source"
            )
        if self._edit_degraded:
            degraded_reasons.append(
                "the editing agent was unavailable; the draft was assembled mechanically"
            )

        verified = (
            bundle is not None
            and not self._edit_degraded
            and not dangling
            and ratio >= self.settings.grounding_threshold
            and entailment.ratio >= self.settings.min_entailment_ratio
            and not entailment.unsupported
            and not int(claim_report.get("unsupported", 0) or 0)
        )
        if verified:
            verdict = "pass"
        degraded_reasons = list(dict.fromkeys(degraded_reasons))
        result = AuditResult(
            grounded_ratio=round(ratio, 3),
            cited_claims=cited_claims,
            total_claims=total_claims,
            unsupported_claims=unsupported,
            dangling_citations=dangling,
            uncited_paragraphs=uncited,
            gaps=gaps,
            verdict=verdict,
            notes=notes,
            entailment_ratio=entailment.ratio,
            entailment_total=entailment.total,
            entailment_supported=entailment.supported,
            entailment_partial=entailment.partial,
            entailment_unsupported=entailment.unsupported,
            entailment_unverified=entailment.unverified,
            claim_verification={
                str(key): int(value) for key, value in claim_report.items() if isinstance(value, int)
            },
            audit_method=method,
            verified=verified,
            degraded=not verified,
            degraded_reasons=degraded_reasons,
        )
        self._log(
            f"Audit: grounding {ratio:.0%}, entailment {entailment.ratio:.0%} "
            f"({entailment.supported}/{entailment.total} supported), "
            f"{len(dangling)} dangling citation(s), {len(gaps)} gap(s) -> "
            f"{'verified' if verified else 'degraded'} ({verdict})"
        )
        if result.degraded:
            self._alert("run_degraded", "audit did not verify the report", reasons=degraded_reasons)
        self._checkpoint(
            "audit",
            "completed",
            {
                "verdict": result.verdict,
                "grounding": result.grounded_ratio,
                "entailment": result.entailment_ratio,
                "verified": result.verified,
                "degraded_reasons": degraded_reasons,
            },
        )
        return result

    def _evaluate(self, report: ResearchReport) -> EvaluationResult | None:
        """Optionally perform one independent, non-mutating LLM quality review."""
        if not self.settings.enable_continuous_evaluation:
            return None
        self._guard("continuous_evaluation")
        self._checkpoint("continuous_evaluation", "running")
        prompt = (
            "You are an independent research-quality judge. Score this report from 0 to 1 "
            "for clarity, evidence discipline, and whether conclusions remain within cited "
            "evidence. Do not invent facts. Return the requested JSON only.\n\n"
            f"Grounding audit: {report.audit.model_dump_json()}\n"
            f"Report (truncated):\n{report.draft_markdown[:12000]}"
        )
        try:
            with usage_scope(agent="judge", track="evaluation"):
                result = self._llm("strong").call(prompt, response_model=EvaluationResult)
            if not isinstance(result, EvaluationResult):
                raise TypeError("judge returned an invalid evaluation")
        except Exception as exc:
            self._checkpoint("continuous_evaluation", "failed", {"error": str(exc)})
            self._trace("evaluation.completed", status="error", error_type=type(exc).__name__)
            return None
        self._checkpoint("continuous_evaluation", "completed", {"score": result.score, "verdict": result.verdict})
        self._trace("evaluation.completed", score=result.score, verdict=result.verdict)
        return result

    # --------------------------------------------------------------- stage 4
    def _research_gaps(self, gaps: list[str]) -> None:
        self._guard("gap_research")
        self._checkpoint("gap_research", "running", {"gaps": len(gaps)})
        llm = self._llm("fast")
        try:
            with usage_scope(agent="gap_researcher", track="gaps"):
                run_gap_research(
                    self.context,
                    llm,
                    self.settings.topic,
                    gaps,
                    verbose=self.settings.crew_verbose,
                )
        except Exception as exc:
            self._log(f"Gap research failed: {exc}")
            self._dead_letter("gap_research", exc, {"topic": self.settings.topic, "gaps": gaps})
            self._checkpoint("gap_research", "failed", {"error": str(exc)})
        else:
            self._checkpoint("gap_research", "completed", {"gaps": len(gaps)})

    def _gap_notes(self, gaps: list[str]) -> str:
        if not gaps:
            return ""
        return "\n".join(f"- {gap}" for gap in gaps)

    def _dead_letter(self, stage: str, error: Exception, payload: dict[str, object]) -> None:
        """Persist failed work for inspection/replay without silently retrying it."""
        if self.context.dead_letters is None:
            return
        path = self.context.dead_letters.enqueue(stage, error, payload)
        self._log(f"Dead-lettered {stage} failure to {path}")

    def _request_degraded_approval(self, reason: str, summary: dict[str, object]) -> None:
        """Escalate partial/fail-closed outcomes to an operator when enabled."""
        self._alert("run_degraded", reason, **summary)
        if not self.settings.require_human_on_degraded:
            return
        self._checkpoint("degraded", "awaiting_approval", {"reason": reason, **summary})
        self.approval.request_approval(
            "degraded",
            self.run_id,
            self.settings.topic,
            {"reason": reason, **summary},
            payload={"state": self.context.run_state.snapshot() if self.context.run_state else {}},
            force=True,
        )
        self._checkpoint("degraded", "approved", {"reason": reason})

    # --------------------------------------------------------------- output
    def _build_report(
        self,
        plan: ResearchPlan,
        track_results: dict[str, TrackResult],
        draft: str,
        audit: AuditResult,
        rounds: int,
    ) -> ResearchReport:
        ordered = self._ordered_results(track_results)
        from .entailment import audit_claims

        finding_audit = audit_claims(
            self.context.evidence,
            [claim for result in ordered for claim in result.findings],
            supported_at=self.settings.entailment_supported_at,
            partial_at=self.settings.entailment_partial_at,
        )
        timeline: list[TimelineEntry] = []
        for result in ordered:
            timeline.extend(result.timeline)
        timeline.sort(key=lambda entry: entry.date)

        open_questions: list[str] = []
        for result in ordered:
            open_questions.extend(result.open_questions)
        open_questions.extend(audit.gaps)

        metrics = self.usage.snapshot()
        metrics.update(self.context.evidence.summary())
        metrics["domain"] = DOMAIN
        metrics["tracks"] = [track.track for track in plan.tracks] or list(self.settings.tracks)
        metrics["grounding_threshold"] = self.settings.grounding_threshold
        metrics["entailment_threshold"] = self.settings.min_entailment_ratio
        covered_tracks = [
            section.track
            for section in ordered
            if not section.warnings
            and (
                section.findings
                or self.context.evidence.claims(track=section.track)
            )
        ]
        metrics["tracks_planned"] = len(plan.tracks) or len(self.settings.tracks)
        metrics["tracks_covered"] = len(covered_tracks)
        metrics["tracks_not_covered"] = [
            section.track for section in ordered if section.track not in covered_tracks
        ]
        metrics["coverage_complete"] = len(covered_tracks) == metrics["tracks_planned"]
        metrics["coverage"] = (
            f"{len(covered_tracks)}/{metrics['tracks_planned']}"
        )
        # One vocabulary across every surface: the ledger records findings, the
        # audit counts the claims that reached the report, and coverage is a
        # fraction of planned tracks. The old ambiguous ``claims`` key conflated
        # all three.
        metrics["findings_recorded"] = len(self.context.evidence.claims())
        metrics["claims_in_report"] = audit.total_claims
        metrics["claims_cited"] = audit.cited_claims
        metrics["claim_verification"] = {
            "total": finding_audit.total,
            "supported": finding_audit.supported,
            "partial": finding_audit.partial,
            "unsupported": finding_audit.unsupported,
            "unverified": finding_audit.unverified,
            "ratio": finding_audit.ratio,
        }
        metrics["audit_status"] = audit.status
        metrics["audit_method"] = audit.audit_method
        metrics["edit_status"] = "degraded" if self._edit_degraded else "completed"
        bundle = active_prompts()
        metrics["prompt_version"] = bundle.version
        metrics["prompt"] = bundle.manifest()
        metrics["models"] = model_manifest(
            {
                "strong": self.settings.model_strong,
                "fast": self.settings.model_fast,
                "planner": self.settings.model_planner,
            },
            fallbacks=tuple(self.settings.model_fallbacks),
        )
        metrics["contracts"] = contract_manifest()
        metrics["cost_drivers"] = _cost_drivers(metrics)
        if self.context.archive is not None:
            metrics["evidence_archive"] = self.context.archive.stats()
        metrics["deterministic_allocations"] = dict(self._allocations)
        metrics["memory_health"] = {
            **self.context.evidence.summary(),
            "state_path": str(self.context.run_state.path) if self.context.run_state else None,
        }

        return ResearchReport(
            topic=self.settings.topic,
            domain=DOMAIN,
            title=self._report_title(),
            executive_summary=self._extract_summary(draft),
            draft_markdown=draft,
            sections=ordered,
            timeline=timeline,
            open_questions=list(dict.fromkeys(open_questions)),
            sources=self.context.evidence.sources(),
            audit=audit,
            metrics=metrics,
            rounds=rounds,
            methodology=self._methodology(plan, ordered),
        )

    def _extract_summary(self, draft: str) -> str:
        marker = "## Executive Summary"
        if marker in draft:
            tail = draft.split(marker, 1)[1]
            for separator in ("\n## ", "\n# "):
                if separator in tail:
                    tail = tail.split(separator, 1)[0]
            return tail.strip()
        return draft.strip()[:1200]

    def _methodology(self, plan: ResearchPlan, results: list[TrackResult]) -> str:
        tracks = plan.tracks or []
        track_names = ", ".join(track.track for track in tracks) or ", ".join(
            result.track for result in results
        )
        return (
            f"Mars is an {DOMAIN}-only research system. It ran "
            f"{len(tracks) or len(results)} parallel research tracks "
            f"({track_names}), each staffed by a "
            "researcher, an evidence analyst and a section writer, followed by an "
            "editor-in-chief and an independent grounding auditor. "
            f"{len(self.context.evidence.sources())} sources were registered; "
            f"{len(self.context.evidence.claims())} findings were recorded in the "
            "claim ledger, and the report cites a verified subset of them. "
            f"Evidence snapshots are content-addressed and archived at "
            f"{self.settings.evidence_dir if self.settings.archive_enabled else 'disabled'}. "
            "Every cited sentence was checked against its archived source text with a "
            "deterministic entailment audit in addition to the LLM auditor."
        )

    def _persist(
        self, report: ResearchReport, markdown: str, plan: ResearchPlan
    ) -> dict[str, str]:
        output_dir = Path(self.settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base = self._artifact_base(report.topic)
        degraded = report.audit.status != "verified"

        paths = {
            # A degraded run must not be mistakable for a finished report.
            "report_md": str(
                base.with_suffix(".DEGRADED.md" if degraded else ".md")
            ),
            "report_json": str(base.with_suffix(".json")),
            "sources_json": str(self._sources_path(report.topic)),
            "plan_json": str(self._plan_path(report.topic)),
            "run_json": str(base.with_name(f"{base.name}-run.json")),
            "report_status": "degraded" if degraded else "verified",
        }

        atomic_write_text(paths["report_md"], markdown)
        atomic_write_json(paths["report_json"], report.model_dump(mode="json"))
        self._persist_sources(report.topic)
        atomic_write_json(paths["plan_json"], plan.model_dump(mode="json"))
        atomic_write_json(
            paths["run_json"],
            {
                "schema_version": 2,
                "run_id": self.run_id,
                "domain": DOMAIN,
                "code_version": _code_version(),
                "lockfile_hash": _lockfile_hash(),
                "prompt_version": active_prompts().version,
                "prompt": active_prompts().manifest(),
                "contracts": contract_manifest(),
                "model_snapshot": {
                    "provider": self.settings.llm_provider,
                    "base_url": self.settings.openrouter_base_url,
                    "strong": self.settings.model_strong,
                    "fast": self.settings.model_fast,
                    "planner": self.settings.model_planner,
                    "fallbacks": list(self.settings.model_fallbacks),
                    "registry": model_manifest(
                        {
                            "strong": self.settings.model_strong,
                            "fast": self.settings.model_fast,
                            "planner": self.settings.model_planner,
                        },
                        fallbacks=tuple(self.settings.model_fallbacks),
                    ),
                },
                "data_disclosure": _disclosure(self.settings),
                "settings": self.settings.to_dict(),
                "metrics": report.metrics,
                "audit": report.audit.model_dump(mode="json"),
                "audit_status": report.audit.status,
                "rounds": report.rounds,
                "http": self.context.http.stats if self.context.http else {},
                "evidence_archive": (
                    self.context.archive.stats() if self.context.archive else {"entries": 0, "bytes": 0}
                ),
                "llm_circuit": self.context.llm_breaker.snapshot() if self.context.llm_breaker else {},
                "llm_budget": self.context.llm_budget.snapshot() if self.context.llm_budget else {},
                "dead_letter_path": str(self.context.dead_letters.path) if self.context.dead_letters else None,
                "state_path": str(self.context.run_state.path) if self.context.run_state else None,
                "trace_path": str(self.context.observer.path) if self.context.observer else None,
            },
        )
        self._log(f"Report written to {paths['report_md']}")
        return paths


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in (text or "report")]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:60] or "report"


def _code_version() -> str:
    """Best-effort build identity for manifests (never fails a run)."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        sha = result.stdout.strip()
        if sha:
            return sha
    except Exception:  # pragma: no cover - git is optional at runtime
        pass
    from . import __version__

    return f"mars-{__version__}"


def _prompt_version() -> str:
    return active_prompts().version


def _cost_drivers(metrics: dict[str, object]) -> dict[str, object]:
    """Name the agent and track that consumed the most cost and tokens.

    Attribution is the point of the per-agent/per-track buckets: when a bill
    spikes, the run manifest should answer "who spent it" without a manual
    join over traces.
    """

    def top(key: str, *, by_tokens: bool) -> dict[str, object] | None:
        buckets = metrics.get(key)
        if not isinstance(buckets, dict) or not buckets:
            return None

        def score(item: tuple[str, object]) -> tuple[float, float]:
            bucket = item[1] if isinstance(item[1], dict) else {}
            tokens = float(bucket.get("prompt_tokens", 0) or 0) + float(
                bucket.get("completion_tokens", 0) or 0
            )
            cost = float(bucket.get("estimated_cost_usd", 0.0) or 0.0)
            return (tokens, cost) if by_tokens else (cost, tokens)

        name, bucket = max(buckets.items(), key=score)
        value = bucket if isinstance(bucket, dict) else {}
        return {
            "name": name,
            "calls": value.get("calls", 0),
            "tokens": int(value.get("prompt_tokens", 0) or 0)
            + int(value.get("completion_tokens", 0) or 0),
            "estimated_cost_usd": value.get("estimated_cost_usd", 0.0),
        }

    return {
        "top_agent_by_cost": top("by_agent", by_tokens=False),
        "top_track_by_cost": top("by_track", by_tokens=False),
        "top_agent_by_tokens": top("by_agent", by_tokens=True),
        "top_track_by_tokens": top("by_track", by_tokens=True),
    }


def _lockfile_hash() -> str | None:
    """Hash the dependency lockfile when one is present (reproducibility)."""
    import hashlib

    for name in ("uv.lock", "poetry.lock", "requirements.lock", "requirements.txt"):
        path = Path(name)
        if path.is_file():
            try:
                return hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                return None
    return None
