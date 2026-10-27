"""Streamlit front-end for Mars.

Run it with::

    streamlit run src/mars/app.py

The UI is a thin layer over the same pipeline the CLI uses: pick a topic, watch
the agents work, read the report, and inspect the evidence that backs it.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from mars.auth import (
    allow_anonymous,
    configured_credentials,
    tenant_slug,
    verify_credentials,
)
from mars.config import DEFAULT_DATA_DIR, ConfigError, build_settings
from mars.domain import DOMAIN, SUBTOPIC_EXAMPLES
from mars.pipeline import ResearchPipeline

# Note: For deployment, configure your reverse proxy (nginx, Cloudflare, etc.)
# with a longer timeout (e.g., proxy_read_timeout 600s) to avoid 504 errors
# on long-running research tasks.

st.set_page_config(
    page_title="Mars - agentic AI research system",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

INTRO = f"""
**Mars** is an **{DOMAIN}-only** multi-agent research system. A research director
splits the subtopic into parallel tracks drawn from a fixed Agentic AI taxonomy;
each track is staffed by a researcher, an evidence analyst and a section writer;
an editor-in-chief assembles the report; and an independent auditor checks every
citation against the source registry and names the gaps that remain.
"""


@dataclass(frozen=True)
class RunConfig:
    """Immutable run configuration for cache key."""
    topic: str
    depth: str
    model: str
    fast_model: str
    require_approval: bool
    approval_checkpoints: tuple[str, ...]
    tenant: str = "anonymous"
    run_id: str = ""
    resume: bool = False


def _tenant_paths(tenant: str) -> dict[str, Path]:
    """Per-tenant directories so one user cannot read another's artefacts."""
    return {
        "output_dir": Path("research_output") / tenant,
        # Keep the whole data dir under the tenant so the derived run-state
        # journal (data_dir/runs) is isolated too.
        "cache_dir": DEFAULT_DATA_DIR / tenant / "cache",
        "dead_letter_dir": DEFAULT_DATA_DIR / "dead_letters" / tenant,
        "observability_dir": DEFAULT_DATA_DIR / "traces" / tenant,
        "approval_dir": Path(".mars_approvals") / tenant,
    }


def _build_run_settings(config: RunConfig):
    """Build settings for one run.

    This used to be wrapped in ``st.cache_resource``, which pinned settings
    (and their API keys and tenant paths) in a process-wide cache across
    sessions.  Settings are cheap to build, so they are constructed fresh.
    """
    overrides: dict[str, Any] = {
        "topic": config.topic,
        "depth": config.depth,
        **_tenant_paths(config.tenant),
    }
    if config.run_id:
        overrides["run_id"] = config.run_id
    if config.model:
        overrides["model_strong"] = config.model
        overrides["model_planner"] = config.model
    if config.fast_model:
        overrides["model_fast"] = config.fast_model
    if config.require_approval:
        overrides["require_human_approval"] = True
        overrides["approval_checkpoints"] = config.approval_checkpoints
    settings = build_settings(**overrides)
    # Fail closed: there is no mock/offline fallback in production.
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Configure it before starting a run."
        )
    try:
        settings.validate()
    except ConfigError as exc:
        raise RuntimeError(f"Invalid configuration: {exc}") from exc
    return settings


def _estimate_cost(depth: str, model: str = "") -> str:
    """Model-aware cost estimate for the run."""
    from mars.metrics import estimate_run_cost

    resolved = model.strip() or "nvidia/nemotron-3-ultra-550b-a55b:free"
    cost, tokens = estimate_run_cost(resolved, depth)
    if cost is None:
        return f"pricing unknown for `{resolved}` (~{tokens:,} tokens)"
    return f"${cost:.2f} (~{tokens:,} tokens on `{resolved}`)"


def _render_sidebar() -> dict[str, Any]:
    """Render sidebar configuration and return the config dict."""
    with st.sidebar:
        st.header("⚙️ Run Configuration")

        topic = st.text_input(
            "Research Topic",
            value="agentic AI",
            help=(
                f"An {DOMAIN} subtopic to investigate. Mars does not research "
                f"anything outside {DOMAIN}."
            ),
        )

        depth = st.selectbox(
            "Research Depth",
            ["quick", "standard", "deep"],
            index=1,
            help="Quick: 1 round, fewer sources. Standard: 2 rounds. Deep: 3 rounds, more sources.",
        )

        with st.expander("🤖 Model Selection", expanded=False):
            model = st.text_input(
                "Editor / Auditor Model",
                value="",
                placeholder="nvidia/nemotron-3-ultra-550b-a55b:free",
                help="Stronger model for planning, editing, and auditing.",
            )
            fast_model = st.text_input(
                "Researcher Model",
                value="",
                placeholder="nvidia/nemotron-3-ultra-550b-a55b:free",
                help="Faster/cheaper model for research and analysis.",
            )

        st.caption(
            f"💰 Estimated cost: **{_estimate_cost(depth, model)}**"
        )

        with st.expander("👤 Human-in-the-loop", expanded=False):
            require_approval = st.checkbox(
                "Require human approval",
                value=False,
                help=(
                    "Pause at configured checkpoints for manual approval. "
                    "Approvals are written to .mars_approvals/ and managed via "
                    "'mars approve' / 'mars reject' CLI commands."
                ),
            )
            approval_checkpoints = st.multiselect(
                "Approval checkpoints",
                ["plan", "draft", "revision"],
                default=["plan", "draft"],
                disabled=not require_approval,
                help="At which pipeline stages to request approval.",
            )

        with st.expander("♻️ Resume a run", expanded=False):
            resume = st.checkbox(
                "Resume an interrupted run",
                value=False,
                help=(
                    "Reuse the persisted plan and completed research tracks for an "
                    "existing run id instead of researching them again."
                ),
            )
            resume_run_id = st.text_input(
                "Run ID",
                value="",
                placeholder="run-20261026-120000-ab12cd34ef",
                disabled=not resume,
                help="Find run ids in the state directory or the run manifest.",
            )
            if resume and not resume_run_id.strip():
                st.warning("A run id is required to resume.")

        st.divider()
        st.subheader("How it works")
        st.markdown(
            """
1. **Director** — splits the topic into tracks and search plans.
2. **Track teams** — researcher → analyst → writer, run in parallel.
3. **Editor** — assembles one coherent, cited report.
4. **Auditor** — verifies citations, flags unsupported claims, names gaps.
5. **Gap round** — targeted researchers close what the auditor found missing.
"""
        )

        return {
            "topic": topic,
            "depth": depth,
            "model": model,
            "fast_model": fast_model,
            "require_approval": require_approval,
            "approval_checkpoints": tuple(approval_checkpoints),
            "resume": resume,
            "run_id": resume_run_id.strip(),
        }


def _render_results(result: dict[str, Any]) -> None:
    """Render the research results in tabs."""
    markdown = result.get("markdown", "")
    report = result.get("report")
    audit = result.get("audit")
    metrics = result.get("metrics", {})

    if audit is not None and getattr(audit, "status", "unreviewed") != "verified":
        reasons = "; ".join(getattr(audit, "degraded_reasons", []) or [])
        st.warning(
            "**Degraded report - not verified.** "
            + (reasons or "The audit did not verify this report.")
            + " The artefact is labelled `.DEGRADED.md`; treat every claim as unverified."
        )

    # Metrics header
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("🎯 Grounding", f"{audit.grounded_ratio:.0%}" if audit else "-")
    with col2:
        st.metric("🗺️ Coverage", metrics.get("coverage", "-"))
    with col3:
        st.metric("📚 Sources", metrics.get("sources", 0))
    with col4:
        st.metric("🔎 Findings", metrics.get("findings_recorded", 0))
    with col5:
        st.metric(
            "🔗 Claims cited",
            f"{metrics.get('claims_cited', 0)}/{metrics.get('claims_in_report', 0)}",
        )
    with col6:
        st.metric("💵 Cost", f"${metrics.get('estimated_cost_usd', 0):.4f}")

    # Tabs for different views
    report_tab, sources_tab, evidence_tab, run_tab = st.tabs(
        ["📄 Report", "📚 Sources", "🔍 Findings", "📊 Run Data"]
    )

    with report_tab:
        st.markdown(markdown)
        # Download button
        report_name = Path(
            result.get("paths", {}).get("report_md") or "mars-report.md"
        ).name
        st.download_button(
            label="📥 Download Report (Markdown)",
            data=markdown,
            file_name=report_name,
            mime="text/markdown",
            use_container_width=True,
        )

    with sources_tab:
        sources = result.get("sources", [])
        st.caption(f"{len(sources)} sources registered during the run.")
        for source in sources:
            with st.expander(f"[{source.id}] {source.title}"):
                st.markdown(f"**URL:** {source.url}")
                st.markdown(f"**Kind:** {source.kind} · **Provider:** {source.provider}")
                if source.published:
                    st.markdown(f"**Published:** {source.published}")
                if source.snippet:
                    st.caption(source.snippet)

    with evidence_tab:
        findings = []
        for section in getattr(report, "sections", []) or []:
            for finding in section.findings:
                findings.append((section.track, finding))
        if not findings:
            st.info("No structured findings were recorded.")
        for track, finding in findings:
            evidence = ", ".join(finding.evidence) or "no evidence"
            # Findings are model/web-derived: render them as plain Markdown,
            # never as raw HTML.
            st.markdown(
                f"- **[{track}]** {finding.text}  \n"
                f"  _evidence: {evidence} · confidence {finding.confidence:.2f} "
                f"· {finding.kind}_"
            )

    with run_tab:
        st.json(metrics)
        paths = result.get("paths", {})
        if paths:
            st.caption(f"Report written to `{paths.get('report_md')}`")
            with st.expander("All output paths"):
                st.json(paths)


def _run_pipeline_background(
    settings,
    log_queue: queue.Queue,
    result_holder: dict[str, Any],
    *,
    resume: bool = False,
) -> None:
    """Background worker function for running the pipeline."""
    def reporter(message: str) -> None:
        log_queue.put(("log", message))

    pipeline = ResearchPipeline(settings, reporter=reporter)
    try:
        outcome = pipeline.run(resume=resume)
        result_holder["markdown"] = outcome.report_markdown
        result_holder["report"] = outcome.report
        result_holder["sources"] = outcome.evidence.sources()
        result_holder["metrics"] = outcome.report.metrics
        result_holder["audit"] = outcome.report.audit
        result_holder["paths"] = outcome.output_paths
        log_queue.put(("done", ""))
    except Exception as exc:  # pragma: no cover - surfaced in the UI
        log_queue.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        pipeline.context.close()
        log_queue.put(("end", ""))


def _check_background_task() -> bool:
    """Check if background task is complete. Returns True if done."""
    worker: threading.Thread | None = st.session_state.get("bg_worker")
    log_queue: queue.Queue | None = st.session_state.get("bg_log_queue")
    result_holder: dict[str, Any] | None = st.session_state.get("bg_result_holder")
    logs: list[str] = st.session_state.get("bg_logs", [])

    if not worker or not worker.is_alive():
        # Thread finished, collect result
        if result_holder:
            st.session_state.result = result_holder or {"error": "run produced no result"}
        else:
            st.session_state.result = {"error": "run produced no result"}
        st.session_state.running = False
        st.session_state.bg_worker = None
        st.session_state.bg_log_queue = None
        st.session_state.bg_result_holder = None
        st.session_state.bg_logs = []
        return True

    # Thread still running, drain log queue
    if log_queue:
        while True:
            try:
                kind, payload = log_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                logs.append(payload)
            elif kind == "error":
                logs.append(f"❌ {payload}")
            elif kind == "done":
                logs.append("✅ Research complete!")
            elif kind == "end":
                pass
        st.session_state.bg_logs = logs

    return False


def _login_gate() -> str:
    """Enforce UI authentication and return the tenant slug.

    The UI can spend money and read reports, so it fails closed when no
    password is configured.  ``MARS_UI_ALLOW_ANONYMOUS=1`` is the explicit
    opt-out for local demos.
    """
    credentials = configured_credentials()
    if credentials is None:
        if allow_anonymous():
            return tenant_slug("anonymous")
        st.error(
            "The Mars UI is locked because `MARS_UI_PASSWORD` is not set. "
            "Set `MARS_UI_PASSWORD` (and optionally `MARS_UI_USERNAME`), or "
            "set `MARS_UI_ALLOW_ANONYMOUS=1` for a local demo."
        )
        st.stop()

    expected_user, expected_password = credentials
    if st.session_state.get("authenticated"):
        return tenant_slug(st.session_state.get("username") or expected_user)

    st.subheader("🔐 Sign in")
    with st.form("mars-login"):
        username = st.text_input("Username", value=expected_user)
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
    if submitted:
        if verify_credentials(username, password, expected_user, expected_password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()


def _render_account(tenant: str) -> None:
    """Show the signed-in tenant and offer a sign-out control."""
    with st.sidebar:
        st.divider()
        st.caption(f"Signed in as **{tenant}**")
        if st.button("Sign out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.result = None
            st.session_state.run_config = None
            st.rerun()


def main() -> None:
    """Streamlit entry point."""
    # Initialize session state
    if "result" not in st.session_state:
        st.session_state.result = None
    if "run_config" not in st.session_state:
        st.session_state.run_config = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "bg_worker" not in st.session_state:
        st.session_state.bg_worker = None
    if "bg_log_queue" not in st.session_state:
        st.session_state.bg_log_queue = None
    if "bg_result_holder" not in st.session_state:
        st.session_state.bg_result_holder = None
    if "bg_logs" not in st.session_state:
        st.session_state.bg_logs = []

    tenant = _login_gate()

    st.title("🛰️ Mars — Multi-Agent Research System")
    st.markdown(INTRO)

    # Check for background task completion (non-blocking)
    if st.session_state.running and st.session_state.bg_worker:
        task_done = _check_background_task()
        if task_done:
            st.rerun()
        else:
            # Show progress and poll with a configurable backoff instead of a
            # hot loop that burns CPU and reruns the whole page.
            _render_progress_ui()
            time.sleep(float(st.session_state.get("ui_poll_seconds", 1.0)))
            st.rerun()

    # Sidebar configuration
    sidebar_config = _render_sidebar()
    _render_account(tenant)

    # Main content area - Topic input prominently displayed
    col_main, col_actions = st.columns([3, 1])
    with col_main:
        topic = st.text_input(
            "🔍 Which Agentic AI subtopic should Mars research?",
            value=sidebar_config["topic"],
            key="main_topic_input",
            placeholder="e.g., agent memory, tool use, multi-agent orchestration...",
            label_visibility="collapsed",
        )
        st.caption(
            f"Mars is domain-locked to **{DOMAIN}**. Try: "
            + ", ".join(SUBTOPIC_EXAMPLES[:4])
            + "."
        )
    with col_actions:
        st.write("")  # spacing
        run_button = st.button(
            "🚀 Run Research",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.running,
        )

    # Sync sidebar topic with main input
    if topic != sidebar_config["topic"]:
        sidebar_config["topic"] = topic

    # Handle run button click - start background task and return immediately
    if run_button and not st.session_state.running:
        if sidebar_config["resume"] and not sidebar_config["run_id"]:
            st.error("Enter the run id you want to resume.")
            st.stop()
        st.session_state.running = True
        st.session_state.result = None
        st.session_state.run_config = RunConfig(
            topic=sidebar_config["topic"],
            depth=sidebar_config["depth"],
            model=sidebar_config["model"],
            fast_model=sidebar_config["fast_model"],
            require_approval=sidebar_config["require_approval"],
            approval_checkpoints=sidebar_config["approval_checkpoints"],
            tenant=tenant,
            run_id=sidebar_config["run_id"],
            resume=sidebar_config["resume"],
        )

        # Build settings fresh for this run.  A missing live key fails closed
        # here rather than silently producing fabricated research.
        try:
            settings = _build_run_settings(st.session_state.run_config)
        except RuntimeError as exc:
            st.session_state.running = False
            st.error(str(exc))
            st.stop()

        st.session_state.ui_poll_seconds = settings.ui_poll_seconds
        log_queue: queue.Queue = queue.Queue()
        result_holder: dict[str, Any] = {}
        # Capture everything the worker needs before the thread starts: Streamlit
        # session state is not safe to read from a non-script thread.
        resume_requested = st.session_state.run_config.resume

        def run_worker():
            _run_pipeline_background(
                settings,
                log_queue,
                result_holder,
                resume=resume_requested,
            )

        worker = threading.Thread(target=run_worker, daemon=True)
        worker.start()

        # Store in session state for polling
        st.session_state.bg_worker = worker
        st.session_state.bg_log_queue = log_queue
        st.session_state.bg_result_holder = result_holder
        st.session_state.bg_logs = []

        st.rerun()

    # Show progress UI if running
    if st.session_state.running:
        _render_progress_ui()

    # Show results if available
    if st.session_state.result:
        result = st.session_state.result
        if "error" in result:
            st.error(result["error"])
        else:
            st.divider()
            _render_results(result)

            # New run button
            st.divider()
            if st.button("🔄 Start New Research", use_container_width=True):
                st.session_state.result = None
                st.session_state.run_config = None
                st.rerun()

    elif not st.session_state.running:
        st.info("👈 Configure your research in the sidebar, then click **Run Research** above.")


def _render_progress_ui() -> None:
    """Render the progress UI for a running background task."""
    logs = st.session_state.get("bg_logs", [])

    with st.status("🔄 Research in progress...", expanded=True):
        if logs:
            st.text("\n".join(logs[-20:]))
        else:
            st.text("Starting research pipeline...")

        # Auto-refresh hint
        st.caption("This page will auto-refresh until complete.")


if __name__ == "__main__":
    main()