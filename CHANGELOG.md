# Changelog

All notable changes to Mars are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
semantic versioning.

## [Unreleased]

Release engineering: the five production gaps from the readiness review are
closed with code, tests and a documented procedure.

### Added

- **Fixed evaluation set.** `evals/agentic_ai_v1.json` holds representative
  Agentic AI tasks, off-domain gate cases and an injection probe.
  `mars/evaluation_set.py` validates the dataset, runs gate cases without a
  model, checks saved reports for structure/citations/grounding and compares a
  new run to a baseline. `mars eval` exposes it; CI runs the gate cases.
- **Prompt versioning and rollback.** `mars/prompt_meta.py` records the version,
  channel, release date and changelog; `mars/agents/prompt_registry.py` loads a
  reviewed override file that can replace protocol blocks or append agent
  instructions without a code deploy. `mars prompts` prints the effective
  bundle and checksum, and the manifest records both.
- **Model lifecycle.** `mars/model_registry.py` pins every allowed model id with
  a status, replacement and context window. Retired models fail validation,
  deprecated models warn, retired fallbacks are dropped, and a retired primary
  is substituted from `MARS_MODEL_FALLBACKS` before the first call. `mars models`
  prints the registry.
- **Agent contracts.** `mars/agents/contracts.py` defines a versioned contract
  for every role (responsibilities, inputs, outputs, schema, failure modes and
  invariants). The pipeline validates profiles against contracts at startup and
  records the contract manifest in every report.
- **Cost attribution and per-track budgets.** `UsageTracker` now keeps
  `by_agent` and `by_track` buckets alongside `by_model` and `by_role`, using a
  `usage_scope` context around every stage. Reports expose `cost_drivers`, and
  `MARS_MAX_TOKENS_PER_TRACK` / `MARS_MAX_COST_USD_PER_TRACK` fail closed.

### Changed

- Prompt version is now `2026-10-27.p4-prompt-registry`; the writer and gap
  researcher use the central citation protocol.
- The run manifest now records `prompt`, `contracts` and
  `model_snapshot.registry`.
- **One reporting vocabulary.** Front matter, metrics, the CLI and the UI now
  report `findings_recorded` (claim ledger), `claims_in_report` (audit total),
  `claims_cited` (audit cited) and `coverage: X/Y`; the ambiguous `claims` key is
  gone, and baseline comparison reads the new key with a legacy fallback.
- **Source hygiene defaults.** `MARS_MIN_SOURCE_QUALITY` now defaults to `0.35`
  and `MARS_MAX_SOURCES_PER_DOMAIN` to `5`, so social/UGC pages and repeated
  pages from one host do not inflate the registry.

### Fixed

- **Gap rounds only run when they add evidence.** If follow-up research records
  no new ledger claim, the run keeps `rounds == 1` instead of paying for a
  second edit and audit that reproduces the same verdict.
- **Prompt-injection-flagged sources are never cited.** Flagged ids are stripped
  from the findings handed to the writer, listed as forbidden in its prompt, and
  rejected by the section citation gate.
- **Writers see the snippets for their own track.** Each writer now receives the
  source ids and snippets its findings depend on, so a citation can be traced to
  text it actually read.
- **Provider rate limits no longer abort a paid-for run.** A burst of `429`
  responses used to open the shared circuit breaker within a second, fail every
  parallel track instantly and raise out of the editorial stage. Now a hard
  free-tier quota is detected and reported as an actionable error without
  retrying; a transient rate limit honours `Retry-After` and equal-jitter
  backoff; the breaker counts one failure per call rather than per attempt and
  can be waited out (`MARS_CIRCUIT_WAIT_SECONDS`); a failed editor degrades to a
  citation-preserving mechanical draft; and a run in which every track failed
  stops with a clean `ProviderUnavailableError` instead of a traceback.

## [0.3.0] - 2026-10-26

Agentic AI domain lock. Mars is no longer a general-purpose research system: the
research domain is fixed to **Agentic AI**, and every layer enforces it.

### Added

- **Fixed domain module.** `mars/domain.py` owns the domain (`Agentic AI`), a
  topic gate (`ensure_domain_topic`), a relevance filter for retrieved material
  (`is_domain_relevant`), query scoping (`scope_query`) and the prompt protocol
  (`domain_protocol`).
- **Canonical track taxonomy.** The Director may only choose from
  `foundations`, `architectures`, `capabilities`, `applications`, `safety`,
  `ecosystem` and `future`. Loose or legacy names are mapped through aliases;
  anything else is dropped before research starts.
- **Startup gate.** `Settings.validate()` rejects an off-domain topic or an
  unknown track, and `Settings.to_dict()` / run manifests record the domain.
- **Retrieval filtering.** Search hits and fetched page content must carry
  Agentic AI signal; off-domain results are rejected with an explicit reason.
  Search, arXiv, Semantic Scholar and Hacker News queries are scoped with
  Agentic AI context when they lack it, and Wikipedia lookups are refused when
  the requested term is off-domain.
- **Domain reporting.** Reports and plans carry a `domain` field; the YAML front
  matter, methodology section, data disclosure and run manifest expose it.
- **CLI/UI surface.** `mars domain` prints the taxonomy, `mars doctor` reports
  the domain lock, and the Streamlit UI labels the input as an Agentic AI
  subtopic with domain examples.
- **Tests.** New `tests/test_domain.py` covers the topic gate, taxonomy mapping,
  plan normalisation, retrieval filtering and the prompt protocol.

### Changed

- Agent profiles and task prompts are written for Agentic AI specialists and
  every prompt carries the domain protocol.
- The fallback plan, report title and methodology are domain-specific.
- Prompt version is now `2026-10-26.p3-agentic-ai-domain`.
- Version bumped to `0.3.0`.

### Migration

- Replace any off-domain `MARS_TOPIC` (for example a news or science topic) with
  an Agentic AI subtopic. Startup now fails with a clear message instead of
  researching outside the domain.
- Persisted runs created before 0.3.0 are normalised onto the canonical tracks
  on resume; an off-domain persisted topic is refused.

## [0.2.0] - 2026-10-26

Production-hardening release. Every P1 and P2 item from the readiness assessment is
addressed with a pragmatic, testable implementation.

### Added

- **Resume and replay.** Each run writes a durable state journal under
  `MARS_CACHE_DIR/../runs/<run_id>.state.json`. `mars research --resume
  --run-id ...` and `mars replay --run-id ...` reuse the persisted plan,
  completed research tracks and source registry, and retry only failed work.
- **Unique run ids and atomic persistence.** Artifact filenames now embed a
  collision-resistant `run-<UTC timestamp>-<uuid>` id. Reports, plans, source
  snapshots, run manifests, caches and state journals are written via
  same-directory temp files plus `fsync`/`os.replace`, so a crash cannot leave
  half-written JSON behind.
- **Cost governance.** `MARS_MAX_TOKENS_PER_RUN` and `MARS_MAX_COST_USD` are
  enforced before every LLM call. Unknown models are reported explicitly
  (`cost_known=false`, `unpriced_models`) instead of silently costing `$0`, and
  `MARS_STRICT_PRICING=1` refuses to start with an unpriced model.
- **Process-wide reliability.** Circuit breakers and outbound rate limits are
  now shared by every run in a worker process, and dead-letter queues expose
  read/count/replay markers.
- **Observability and health.** Trace sampling (`MARS_TRACE_SAMPLE_RATE`),
  structured Python logging, optional OTLP export (`MARS_OTEL_ENDPOINT`),
  best-effort `run_failed` / `run_degraded` webhook alerts
  (`MARS_ALERT_WEBHOOK_URL`), and a stdlib health server with `/healthz`,
  `/readyz` and Prometheus `/metrics` (`mars serve`, `mars health`,
  `mars metrics`).
- **Secret-manager integration.** Every secret variable accepts a `<NAME>_FILE`
  companion, so Docker/Kubernetes secret volumes work without adding a cloud
  SDK; `.env` remains supported for local development.
- **Retention.** `mars prune [--days N] [--dry-run]` applies
  `MARS_RETENTION_DAYS` to reports, traces, dead letters, state journals,
  approvals and caches; HTTP and LLM caches are additionally bounded by size
  and entry count with LRU eviction.
- **Startup validation.** `Settings.validate()` fails fast on missing keys,
  impossible limits, bad URLs, unknown providers and strict-pricing violations;
  the CLI and pipeline surface a clean `Configuration error` exit.
- **Deployment and CI.** `Dockerfile`, `docker-compose.yml` (UI plus health
  sidecar), `.dockerignore`, and a GitHub Actions workflow that lints, runs the
  hermetic test suite and builds the image.
- **Test hardening.** New coverage for atomic storage, validation, budgets,
  shared breakers/rate limits, cache eviction, retention, health endpoints,
  resume/replay, concurrency, chaos/fault injection, and an opt-in `live`
  provider contract test.
- **Entailment auditing (P2).** Every substantive cited sentence and every
  recorded claim is checked against the archived text of its cited sources. A
  report is marked `verified` only when the LLM audit and the deterministic
  entailment audit both pass; otherwise it is `degraded`, with the reasons in
  the report appendix and an optional approval gate. Per-claim status is
  persisted on the claim ledger.
- **Immutable evidence snapshots (P2).** Fetched pages and tool results are
  written to a content-addressed (SHA-256), write-once, read-only archive.
  `Source` records the hash, archive path, final URL, content type, HTTP status,
  quality, licence, paywall flag and injection flags. `mars verify <report>`
  re-audits a persisted report offline against the archive.
- **Crawler governance (P2).** `robots.txt` is honoured by default, with
  configurable domain allow/block lists, URL patterns, minimum source quality,
  maximum source age, paywall blocking and licence detection. Rejected search
  hits are reported instead of silently registered.
- **Prompt-injection hardening (P2).** External text is stripped of instruction
  overrides, role changes, prompt-exfiltration and tool directives, hidden HTML
  and control characters, then wrapped in `<untrusted_web_content>` boundaries.
  Every agent prompt carries an explicit untrusted-data protocol.
- **Data governance (P2).** Secret and PII redaction runs before prompts leave
  the process. Optional Fernet encryption at rest covers state journals, dead
  letters and approval payloads. `mars disclosure` documents every external data
  flow and `mars delete-run` removes all artefacts for one run without touching
  the shared evidence archive.
- **Reproducible manifests (P2).** Run manifests (schema v2) record the prompt
  version, lockfile hash, model snapshot, data-disclosure metadata and evidence
  archive statistics alongside the existing code version and run id.

### Changed

- Streamlit settings are built per run instead of being pinned in
  `st.cache_resource`, tenant data directories are fully isolated, progress
  polling backs off (`MARS_UI_POLL_SECONDS`) instead of hot-looping, and the UI
  can resume an interrupted run by id.
- Run manifests include `schema_version`, `run_id` and the source `code_version`
  (git SHA or package version); P2 raises the manifest schema to v2 and adds the
  prompt version, lockfile hash, model snapshot, disclosure and archive stats.
- The audit merges the LLM verdict with deterministic entailment and claim
  verification. Reports and front matter now expose `audit_status`,
  `entailment_score` and the reasons a report is not marked verified.
- Findings and web-derived text are rendered as plain Markdown in the UI; raw
  HTML is never passed to `unsafe_allow_html`.
- Settings and manifest writes redact credentials by default.
- Package version bumped to `0.2.0`.

### Removed

- The deterministic `MockLLM` provider, the `opencode` provider, the curated
  corpus and every offline/mock fallback were removed from `src/mars`.
  `MARS_LLM_PROVIDER` accepts only `openrouter`, and a missing key fails closed.
- `mars research --offline`, `mars replay --offline` and `mars corpus` were
  removed. Test-only deterministic providers and corpus tools now live under
  `tests/support/` and are injected by `tests/conftest.py`.

### Security

- Credentials are redacted from `Settings.to_dict()` and from all persisted
  manifests; historical manifests were scrubbed during the P0 pass.
- The UI fails closed without `MARS_UI_PASSWORD`, approvals are HMAC-signed and
  bounded, and outbound HTTP is restricted to public hosts with SSRF guards.
- External text is treated as untrusted data: prompt-injection payloads are
  neutralised and enclosed in explicit data boundaries.
- Credentials and personal data are redacted from prompts, dead-letter payloads
  and approval artefacts; state and payloads can be encrypted at rest.
- Evidence is archived write-once and re-verifiable, so a report's citations can
  be checked later even if the original pages change or disappear.
