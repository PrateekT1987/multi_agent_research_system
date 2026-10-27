# Mars 🛰️

**Mars** is a multi-agent research system with a single, fixed research domain: **Agentic AI** — AI systems that plan, reason, use tools, remember context and act, from a single LLM agent loop to multi-agent orchestration. It produces a cited, audit-checked report on a subtopic inside that domain.

A research director decomposes the subtopic into tracks from a **fixed Agentic AI taxonomy**. Each track is staffed by a small team that runs in parallel: a **researcher** (live web, academic papers, encyclopaedia, practitioner sources), an **evidence analyst** (turning a raw dossier into structured findings with explicit source ids), and a **section writer**. An **editor-in-chief** assembles the sections into one report, and an independent **grounding auditor** checks every citation against the source registry, flags unsupported claims and names the gaps that remain. If the audit fails, targeted researchers go back out and close those gaps, and the report is revised.

```
plan ──▶ ┌─ track 1 ─┐
         ├─ track 2 ─┼─▶ analyse ─▶ write ─┐
         ├─ track 3 ─┤                     │
         └─ track N ─┘                     ▼
                                  edit ─▶ audit ─▶ (gaps?) ─▶ edit ...
```

The Director chooses a subset of this canonical taxonomy for each subtopic:

| Track | Covers |
|---|---|
| `foundations` | Symbolic agents, RL agents, ReAct and the modern LLM agent loop |
| `architectures` | Planning, memory, tool use, function calling and multi-agent orchestration |
| `capabilities` | What agents can do, benchmarks, evaluation and failure modes |
| `applications` | Production deployments, changed workflows and measured impact |
| `safety` | Prompt injection, security, reliability, oversight and governance |
| `ecosystem` | Frameworks, protocols (MCP, A2A), vendors, adoption and economics |
| `future` | Credible trajectories, open problems and signals to watch |

Off-domain topics are rejected before any model call. `mars domain` prints the
full taxonomy; `src/mars/domain.py` is the single source of truth for the topic
gate, track aliases, retrieval relevance filter and prompt protocol.

---

## Quick start

```bash
# install (uses uv; there is a pyproject.toml with everything you need)
uv sync --extra dev

# live: needs an OpenRouter key
export OPENROUTER_API_KEY=sk-or-v1-...
uv run mars research --topic "agent memory" --depth standard
```

The report (Markdown + JSON), the source registry, the plan and the run manifest are written to `research_output/` (or `--output-dir`). There is no mock or offline provider in production: a missing key fails closed instead of producing fabricated research.

### Streamlit UI

```bash
uv run mars-ui
# or
uv run streamlit run src/mars/app.py
```

The UI is a full-featured browser dashboard:

| Feature | Description |
|---------|-------------|
| **Research Configuration** | Topic input, depth (quick/standard/deep), model selection |
| **Human-in-the-loop** | Optional approval gates at plan/draft/revision checkpoints |
| **Live Progress** | Streaming agent logs with auto-refresh (no 504 timeouts) |
| **Report View** | Searchable/filterable markdown with citation highlighting |
| **Visualizations** | Source type breakdown, grounding score charts (Plotly) |
| **History Browser** | Browse past runs from `research_output/` with metrics comparison |
| **Export** | Download Markdown, generate PDF, copy for Notion/Google Docs |
| **Sources & Findings** | Expandable source cards, structured findings with evidence links |

**Run it:**
```bash
uv run mars-ui
```
Then open http://localhost:8501 in your browser.

---

## Architecture

| Layer | Module | Responsibility |
|---|---|---|
| Orchestration | `mars/pipeline.py` | Stage order, parallel tracks, gap rounds, artefact persistence |
| Domain | `mars/domain.py` | Fixed Agentic AI domain, topic gate, canonical tracks, relevance filter, prompt protocol |
| Agents | `mars/agents/` | CrewAI personas (`profiles.py`), prompt templates (`tasks.py`), crew factory (`factory.py`) |
| Prompt release | `mars/prompt_meta.py`, `mars/agents/prompt_registry.py` | Versioned prompt bundle, dev/staging/prod channels, reviewed override file and per-run checksum |
| Contracts | `mars/agents/contracts.py` | Versioned per-agent interface contracts: responsibilities, inputs, outputs, failure modes and invariants |
| Model lifecycle | `mars/model_registry.py` | Pinned model ids, deprecation/retirement status, replacement and explicit fallback chain |
| Evaluation | `mars/evaluation_set.py`, `evals/` | Fixed regression dataset, domain-gate/injection probes, report assertions and baseline comparison |
| Models | `mars/llm.py` | `OpenRouterLLM` (a `crewai.BaseLLM` over `langchain-openrouter`); prompts are redacted before disclosure |
| Tools | `mars/tools/` | Search, fetch, arXiv, Semantic Scholar, Wikipedia, Hacker News, findings ledger; every result is governed, sanitised and archived |
| Evidence | `mars/evidence.py` | Thread-safe source registry + claim ledger; citation extraction, grounding scoring and entailment checks |
| Archive | `mars/archive.py` | Write-once, content-addressed (SHA-256) evidence snapshots for later re-verification |
| Governance | `mars/governance.py` | robots.txt, domain allow/block lists, quality/recency filters, paywall and licence detection |
| Sanitisation | `mars/sanitize.py` | Prompt-injection neutralisation and explicit `<untrusted_web_content>` boundaries |
| Privacy | `mars/redaction.py`, `mars/crypto.py` | Secret/PII redaction before prompts leave the host; optional Fernet encryption at rest |
| Verification | `mars/verification.py` | Offline re-audit of a persisted report against its archived sources (`mars verify`) |
| Net | `mars/net.py` | `httpx` client with disk cache, sliding-window rate limit, retry with backoff |
| Resilience | `mars/reliability.py` | Shared LLM circuit breaker, jittered bounded retries, run call budget, watchdogs, checkpoint journal and DLQ |
| Output | `mars/reporting.py`, `mars/cli.py`, `mars/app.py` | Report rendering, CLI, Streamlit UI |
| Visualization | `mars/app.py` | Plotly charts for source types, grounding trends |
| Export | `mars/app.py` | PDF (markdown2 + weasyprint), Markdown for Notion/Google Docs |

### Design notes

* **The pipeline owns the control flow.** Each stage is a single-agent CrewAI crew whose prompt is fully materialised before kickoff, so you can always ask "what exactly did the writer see?". CrewAI still does the heavy lifting: native function calling, tool execution and structured output.
* **Citations are ids, not prose.** Every source is registered as `[S1]`, `[S2]`, … Agents may only cite ids they were handed; the auditor cross-checks the report against the registry and reports *dangling* citations explicitly.
* **Everything is recorded.** Sources, findings, the plan, the audit and the run metrics are all serialised next to the report, so a run can be inspected and re-run after the fact.
* **Degrades instead of guessing.** A failed track, an unparseable plan or an unavailable auditor is recorded as a warning or a gap, never papered over with invented text.
* **Fail together, recover deliberately.** All agents in a run share one LLM circuit breaker and one call budget. A provider outage opens the circuit for the fleet, rather than letting parallel tracks amplify retries and spend.
* **Bounded fan-out.** The director's plan is normalised to unique tracks, capped by `MARS_MAX_TRACKS`, then assigned to stable worker lanes. Completion order cannot alter report order or task ownership.
* **Durable supervision.** Stage transitions are atomically checkpointed under `~/.cache/mars/runs`; failed work is appended to the DLQ. Watchdogs bound total run and individual-track time, while request timeouts bound a non-cooperative provider call.
* **Domain-locked by design.** There is no general-purpose mode. The topic gate, the fixed track taxonomy, the prompt protocol and retrieval filtering all enforce Agentic AI; an off-domain topic or source is refused rather than researched.
* **Prompts are a versioned release.** The instruction text is code-reviewed, but the active bundle carries a version, a channel and a checksum. A reviewed override file can replace a protocol block or add agent-specific instructions without a code deploy, and `mars prompts` shows exactly what is active.
* **Models are pinned, not floated.** Every configured model id has a lifecycle status and an explicit replacement. Retired models fail validation, deprecated models warn, and a fallback chain is recorded in the run manifest so a migration is deliberate rather than silent.
* **Agents have contracts.** Each role has a versioned contract describing its inputs, outputs, structured schema, failure modes and invariants. The pipeline refuses to start if a profile has no contract or a contract is inconsistent, and the contract manifest is written into every report.
* **Regressions are measured.** `evals/agentic_ai_v1.json` fixes representative tasks, off-domain refusals and injection probes. `mars eval` runs the gate cases and checks saved reports against the same assertions, with an optional baseline comparison for grounding, entailment, cost and finding count.
* **Spend is attributable.** Token and cost buckets are kept per model, role, agent and track, so a bill spike names the agent and the track that caused it; per-track ceilings fail closed at the next call.

---

## The evidence model

```
Source   id=[S1] url=... title=... kind=web|paper|wiki|community
         content_hash=<sha256> archive_path=... final_url=... http_status=...
         quality=0.85 license=CC-BY paywalled=false injection_flags=[]
Claim    id=C1 text="..." evidence=[S1, S4] confidence=0.8 kind=fact|trend|forecast|controversy
         verification=supported|partial|unsupported|unverified verification_score=0.82
```

* `EvidenceStore` is the single source of truth, shared by all parallel tracks (it is guarded by a lock).
* `grounding_score(text)` → share of prose paragraphs that carry at least one citation.
* `dangling_citations(text)` → citations that point at an unregistered source.
* `uncited_paragraphs(text)` → assertions with no citation at all.
* `entailment_audit(text)` / `verify_claims()` → deterministic lexical checks that each cited sentence or ledger claim is actually supported by the archived text of the sources it cites.
* `render_reference_list()` → the `[S1] Title <url> (kind, provider)` block used in the report's reference section.

---

## Configuration

Everything is driven by environment variables (a `.env` file in the working directory is loaded automatically) and can be overridden per run.

Secrets can be supplied directly or mounted as files: every secret variable also accepts a `<NAME>_FILE` companion (for example `OPENROUTER_API_KEY_FILE=/run/secrets/openrouter`). This works with Docker secrets, Kubernetes secret volumes and most secret managers without adding a cloud SDK.

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | – | Required for the live `openrouter` provider. Mars fails closed rather than fabricating research. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Override for proxies/gateways. |
| `MARS_TOPIC` | `agentic AI` | Default subtopic. Must be inside the fixed Agentic AI domain. |
| `MARS_LLM_PROVIDER` | `openrouter` | Only `openrouter` is supported; any other value fails validation. |
| `MARS_ALLOW_NETWORK` | `true` | `false` = refuse every outbound HTTP request and withhold network tools. |
| `MARS_ALLOW_PRIVATE_NETWORK` | `false` | Allow loopback/private destinations. Local development only; never enable in production. |
| `MARS_MAX_RESPONSE_BYTES` | `10000000` | Hard cap on any HTTP response body. |
| `MARS_HTTP_MAX_REDIRECTS` | `5` | Maximum redirects; every hop is re-validated against the egress policy. |
| `MARS_RESPECT_ROBOTS` | `true` | Honour `robots.txt` before fetching a page. |
| `MARS_ROBOTS_FAIL_CLOSED` | `false` | Treat an unreachable `robots.txt` as a denial. |
| `MARS_ALLOWED_DOMAINS` | – | Comma-separated allowlist; when set, only these domains may be fetched. |
| `MARS_BLOCKED_DOMAINS` | – | Comma-separated blocklist, applied before the allowlist. |
| `MARS_BLOCKED_URL_PATTERNS` | – | Comma-separated regular expressions for URLs to refuse. |
| `MARS_MIN_SOURCE_QUALITY` | `0.35` | Reject search hits below this 0-1 reputation score (official/peer-reviewed rank highest; social/UGC pages fall below the default). |
| `MARS_MAX_SOURCES_PER_DOMAIN` | `5` | Cap how many distinct pages one host may contribute to a run; `0` disables the cap. |
| `MARS_MAX_SOURCE_AGE_DAYS` | – | Reject sources older than this when a publication date is known. |
| `MARS_BLOCK_PAYWALLED` | `false` | Refuse pages whose text advertises a paywall. |
| `MARS_ARCHIVE_ENABLED` | `true` | Write immutable, content-addressed evidence snapshots. |
| `MARS_ARCHIVE_DIR` | `<data dir>/evidence` | Snapshot root (SHA-256 addressed, read-only). |
| `MARS_ARCHIVE_MAX_CHARS` | `2000000` | Maximum characters archived per snapshot; truncation is recorded. |
| `MARS_REDACT_PII` | `true` | Redact emails, phone numbers, SSNs and card numbers from prompts and payloads. |
| `MARS_ENCRYPTION_KEY` | – | Optional Fernet key (or high-entropy string) for encryption at rest. |
| `MARS_ENCRYPT_STATE` | `false` | Encrypt run state, dead letters and approval payloads with `MARS_ENCRYPTION_KEY`. |
| `MARS_ENTAILMENT_SUPPORTED_AT` | `0.55` | Lexical score at which a cited sentence counts as supported. |
| `MARS_ENTAILMENT_PARTIAL_AT` | `0.30` | Score at which a cited sentence counts as partially supported. |
| `MARS_MIN_ENTAILMENT_RATIO` | `0.55` | Minimum entailment ratio required before a report is marked verified. |
| `MARS_MODEL_STRONG` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Editor + auditor model. |
| `MARS_MODEL_FAST` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Researcher + analyst model. |
| `MARS_MODEL_PLANNER` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Director model. |
| `MARS_MODEL_FALLBACKS` | – | Comma-separated ordered fallback model ids. Retired entries are dropped, a retired primary is substituted before the first call, and the usable chain is recorded in the manifest. |
| `MARS_STRICT_MODEL_REGISTRY` | `false` | `true` refuses to start when any configured model is not pinned in `mars/model_registry.py`. |
| `MARS_PROMPT_VERSION` | `2026-10-27.p4-prompt-registry` | Prompt bundle version recorded in every report and manifest. |
| `MARS_PROMPT_CHANNEL` | `prod` | `dev`, `staging` or `prod`; non-`prod` runs emit a startup warning. |
| `MARS_PROMPT_OVERRIDES` | – | Path to a reviewed JSON prompt override (protocols and agent suffixes). Enables rollback without a code deploy. |
| `MARS_DEPTH` | `standard` | `quick` (1 round), `standard` (2), `deep` (3). |
| `MARS_OUTPUT_DIR` | `./research_output` | Where artefacts are written. |
| `MARS_CACHE_DIR` | `~/.cache/mars/cache` | HTTP + LLM cache. |
| `MARS_STATE_DIR` | `<cache parent>/runs` | Durable run-state journals used by resume/replay. |
| `MARS_CACHE_MAX_BYTES` / `MARS_CACHE_MAX_ENTRIES` | `500000000` / `50000` | Hard bounds on the HTTP cache; oldest entries are evicted first. |
| `MARS_LLM_CACHE_TTL_HOURS` / `MARS_LLM_CACHE_MAX_ENTRIES` | `168` / `10000` | LLM response cache freshness and entry cap. |
| `MARS_MAX_LLM_CALLS_PER_RUN` | `160` | Hard provider-call ceiling shared by every agent in one run. |
| `MARS_MAX_TOKENS_PER_RUN` | – | Optional token ceiling; the next call fails once it is spent. |
| `MARS_MAX_COST_USD` | – | Optional USD ceiling enforced before every LLM call. |
| `MARS_MAX_TOKENS_PER_TRACK` | – | Optional per-track token ceiling; the next call inside that track fails closed. |
| `MARS_MAX_COST_USD_PER_TRACK` | – | Optional per-track USD ceiling, attributed by the track scope around each agent call. |
| `MARS_STRICT_PRICING` | `false` | `true` refuses to start when a configured model has no price entry. |
| `MARS_OBSERVABILITY_DIR` | `~/.cache/mars/traces` | JSONL trace directory. |
| `MARS_TRACE_SAMPLE_RATE` | `1.0` | Fraction of runs that emit traces (0-1). |
| `MARS_OTEL_ENDPOINT` | – | Optional OTLP/HTTP JSON endpoint for trace export. |
| `MARS_ALERT_WEBHOOK_URL` | – | Optional JSON webhook for `run_failed` / `run_degraded` alerts. |
| `MARS_SHUTDOWN_GRACE_SECONDS` | `10` | Grace period advertised to container runtimes. |
| `MARS_RETENTION_DAYS` | `30` | Retention window applied by `mars prune`. |
| `MARS_UI_POLL_SECONDS` | `1.0` | Streamlit progress-poll backoff. |
| `MARS_MAX_LLM_RETRIES` | `2` | Retry attempts after an initial transient LLM failure. |
| `MARS_RETRY_BASE_SECONDS` / `MARS_RETRY_MAX_SECONDS` | `1` / `20` | Equal-jitter exponential-backoff range. `Retry-After` is honoured. |
| `MARS_CIRCUIT_FAILURE_THRESHOLD` / `MARS_CIRCUIT_RECOVERY_SECONDS` | `4` / `30` | Shared LLM circuit breaker policy. |
| `MARS_CIRCUIT_WAIT_SECONDS` | `40` | How long one call waits for the shared breaker to recover; `0` fails fast. |
| `MARS_MAX_RUN_SECONDS` / `MARS_MAX_TRACK_SECONDS` | `1800` / `480` | Supervisor watchdog limits. |
| `MARS_MAX_TRACKS` | `6` | Maximum model-generated research tracks accepted per run. |
| `MARS_DEAD_LETTER_DIR` | `~/.cache/mars/dead_letters` | Durable JSONL queue for failed work. |
| `MARS_REQUIRE_HUMAN_ON_DEGRADED` | `false` | Force approval before continuing after track or audit degradation. |
| `MARS_REQUIRE_HUMAN_APPROVAL` | `false` | Enable human gates at the configured checkpoints. |
| `MARS_APPROVAL_CHECKPOINTS` | `plan,draft` | Comma-separated checkpoint names. |
| `MARS_APPROVAL_SECRET` | – | **Required** when approvals are enabled. Approval/rejection files must carry a valid HMAC signature. |
| `MARS_APPROVAL_TIMEOUT_SECONDS` | `1800` | Bounded wait for a human decision; the run fails instead of hanging forever. |
| `MARS_APPROVAL_DIR` | `./.mars_approvals` | Where approval requests and signed decisions are stored. |
| `MARS_UI_PASSWORD` | – | Required to start the Streamlit UI. |
| `MARS_UI_USERNAME` | `admin` | UI login username. |
| `MARS_UI_ALLOW_ANONYMOUS` | `false` | Explicit opt-out of the UI login gate for local demos only. |
| `MARS_VERBOSE` / `MARS_CREW_VERBOSE` | `true` / `false` | Pipeline logging / CrewAI's own console output. |
| `BRAVE_API_KEY`, `SERPER_API_KEY`, `TAVILY_API_KEY`, `EXA_API_KEY` | – | Optional keyed search providers. Keyless DuckDuckGo is used only when listed in `MARS_SEARCH_PROVIDERS`. |

---

## CLI

```bash
mars research --topic "multi-agent orchestration" --depth deep --rounds 3   # full programme
mars research --resume --run-id run-20261026-120000-ab12cd34ef      # resume an interrupted run
mars replay --run-id run-20261026-120000-ab12cd34ef                 # replay a failed run's work
mars doctor --offline                                               # config check, no requests
mars domain                                                         # show the fixed taxonomy
mars models                                                         # pinned models + lifecycle status
mars prompts                                                        # active prompt version, channel, checksum
mars eval --dataset evals/agentic_ai_v1.json                        # run the fixed regression set
mars verify research_output/multi-agent-orchestration-<run-id>.json  # re-audit against snapshots
mars delete-run --run-id run-20261026-120000-ab12cd34ef             # delete one run's artefacts
mars disclosure                                                     # what leaves this host, and where
mars show research_output/multi-agent-orchestration-<run-id>.json    # inspect a stored run
mars health                                                         # liveness/readiness report
mars metrics                                                        # Prometheus text metrics
mars serve --port 8080                                              # /healthz, /readyz, /metrics
mars prune --days 30 --dry-run                                      # retention preview
```

Artifact names embed the run id, so a resumed or replayed run updates its own
files instead of creating a second, ambiguous timestamped set.

## Operations

### Resume and replay

Every run writes an atomic state journal to
`<cache parent>/runs/<run_id>.state.json` and persists the plan, the source
registry and each completed track's result. `--resume` reuses all of that and
only re-runs work that did not finish. `replay` additionally reads the durable
dead-letter queue, records a replay marker, and resumes the same run id.

A run that already completed refuses to resume unless you pass `--force` to
`mars replay`.

### Health and metrics

`mars serve` exposes:

| Endpoint | Purpose |
|---|---|
| `/healthz` | `200` unless a critical check fails (`unhealthy`), `503` otherwise. |
| `/readyz` | `200` only when critical dependencies (configuration, writable data/output dirs) are ready. |
| `/metrics` | Prometheus text exposition: liveness, readiness, cache size/entries, dead-letter depth, disk free, run-state counts and circuit-breaker state. |

`mars health` and `mars metrics` print the same data locally.

### Retention

`mars prune` applies `MARS_RETENTION_DAYS` to reports, traces, dead letters,
state journals, approvals and caches. HTTP and LLM caches are also bounded by
size and entry count with LRU eviction, so an unattended deployment cannot grow
without limit. Use `--dry-run` to preview.

### Deployment

```bash
docker compose up --build          # Streamlit UI on :8501, probes on :8080
docker build -t mars:0.3.0 .
docker run --env-file .env -p 8501:8501 -v mars-data:/data/mars mars:0.3.0
```

The image runs as an unprivileged user, keeps all durable state under
`/data/mars`, and declares a Streamlit health check. CI (`.github/workflows/ci.yml`)
lints, runs the hermetic suite and builds the image on every push.

## Release engineering

The following are release artefacts, not documentation written after the fact.
Each has a command and a manifest field so a run can be tied to the exact
evaluation set, prompt, model and contract that produced it.

### Evaluation set and regression testing

`evals/agentic_ai_v1.json` is the fixed regression set. It mixes representative
Agentic AI tasks, domain-gate refusals and an injection probe. Gate cases run
without a model; research cases are checked against saved report JSON files for
canonical tracks, sources, findings, `[S#]` citations and a minimum grounding
ratio.

```bash
mars eval --dataset evals/agentic_ai_v1.json
mars eval --dataset evals/agentic_ai_v1.json \
  --reports research_output \
  --baseline research_output/agentic-ai-<previous-run>.json
```

The command exits non-zero on a failed case or a regression: grounding or
entailment falling by more than `--max-drop` (default 0.05), estimated cost
rising by more than 25%, tokens by more than 35%, or the finding count
decreasing. The release procedure is in `evals/README.md`: run the set before a
change, run it again after, and only relax a case in a reviewed commit with the
reason recorded.

### Prompt versioning and rollback

Prompt text lives in `mars/agents/profiles.py` and `mars/agents/tasks.py` and is
reviewed like code. Its release metadata lives in `mars/prompt_meta.py`:
`PROMPT_VERSION`, a channel, a release date and a changelog. Every report and
run manifest records the version, channel, checksum and source.

For a change that must not wait for an image rebuild, point
`MARS_PROMPT_OVERRIDES` at a reviewed JSON file:

```json
{
  "prompt_version": "2026-10-27.p4-tighter-citations",
  "channel": "prod",
  "protocols": {
    "citation": "...",
    "untrusted": "...",
    "domain": "..."
  },
  "agent_suffixes": {
    "writer": "Prefer primary sources over surveys."
  }
}
```

`mars prompts` prints the effective bundle and checksum. Rollback is changing
the path back to the previous file and restarting; no code deploy is involved.
An override that disagrees with `MARS_PROMPT_VERSION` or `MARS_PROMPT_CHANNEL`,
names an unknown agent, or leaves a protocol empty fails validation at startup.

### Model pinning and lifecycle

`mars/model_registry.py` is the curated list of allowed model ids. Each pin has
a status (`active`, `deprecated`, `retired`), a release date, an optional
deprecation date, a replacement and a context window. `mars models` prints the
registry and validates the configured roles.

* A **retired** model fails `Settings.validate()`; it is never used silently.
* A **deprecated** model warns at startup and names its replacement.
* An **unpinned** model is allowed by default for custom OpenRouter routes but
  warns; `MARS_STRICT_MODEL_REGISTRY=1` turns that into a startup error.
* `MARS_MODEL_FALLBACKS` is the ordered fallback policy. Retired entries are
  dropped, the usable chain is recorded in the run manifest, and an all-retired
  chain is a hard error. If a configured primary is retired and an active
  fallback exists, the pipeline substitutes it before the first call and records
  the model that actually ran.

Migration procedure: add the new pin and its release metadata, run `mars models`,
run the evaluation set against the new model, compare with `--baseline`, then
change the default and record the old pin's deprecation/replacement. The run
manifest's `model_snapshot.registry` is the audit trail.

### Agent contracts

`mars/agents/contracts.py` defines a versioned contract for every role:
responsibilities, inputs, outputs, structured-output schema, failure modes and
invariants. `validate_contracts()` runs at pipeline startup and refuses to start
if a profile has no contract, a contract has no version, or it names an unknown
schema. `tests/test_contracts.py` keeps the profiles and contracts in sync.
Every report and run manifest carries the full contract manifest.

### Cost attribution and per-agent budgets

`UsageTracker` keeps `by_model`, `by_role`, `by_agent` and `by_track` buckets.
The pipeline wraps each stage in a `usage_scope`, so parallel tracks cannot
attribute each other's calls. Every report's metrics include `by_agent`,
`by_track` and `cost_drivers`, which name the top agent and track by cost and by
tokens. `MARS_MAX_TOKENS_PER_TRACK` and `MARS_MAX_COST_USD_PER_TRACK` fail closed
at the next call inside the offending track, without penalising the others.

## Library use

```python
from mars import ResearchPipeline, build_settings

settings = build_settings(topic="agent memory", openrouter_api_key="sk-or-v1-...")
result = ResearchPipeline(settings).run()

print(result.report_markdown)
print(result.report.audit.grounded_ratio, result.report.audit.entailment_ratio)
print(result.report.audit.status, len(result.evidence.sources()))
```

---

## Quality, governance and data protection

The P2 pass replaced citation-presence scoring with evidence-level checks and made every external data flow explicit.

* **Entailment auditing.** Every substantive cited sentence is checked against the archived text of the sources it cites, and every recorded claim gets a persisted `supported` / `partial` / `unsupported` / `unverified` status. The LLM auditor supplies semantic judgement; the deterministic lexical check supplies a reproducible floor. A report is marked `verified` only when both the LLM audit and the entailment audit pass. Otherwise it is `degraded`, with the reasons printed in the report appendix, and (when configured) routed to a human approval gate.
* **Immutable evidence snapshots.** Fetched pages and tool results are written to a content-addressed (SHA-256), write-once, read-only archive. The hash, final URL, content type, HTTP status, quality and licence are recorded on each `Source`, so `mars verify <report.json>` can re-audit a report months later without touching the network.
* **Crawler governance.** `robots.txt` is honoured by default, and operators can restrict runs with domain allow/block lists, URL patterns, a minimum source-quality score, a maximum source age and a paywall block. Search hits that fail the policy are reported as rejected instead of being silently registered.
* **Prompt-injection hardening.** External text is stripped of instruction-like payloads (role overrides, prompt exfiltration, tool directives, fake role tags, hidden HTML/control characters) and wrapped in `<untrusted_web_content>` boundaries. Agent prompts include an explicit rule that content inside those boundaries is data, never instructions.
* **Data governance.** Prompts are redacted for credentials and (optionally) personal data before they leave the process. State journals, dead letters and approval payloads can be encrypted at rest with `MARS_ENCRYPTION_KEY` + `MARS_ENCRYPT_STATE=1`. `mars disclosure` lists exactly what is sent to the model provider, and `mars delete-run --run-id ...` removes every artefact a run produced without touching the shared evidence archive.
* **Safe rendering.** Findings and web-derived text are rendered as plain Markdown in the Streamlit UI; raw HTML from the model or the web is never passed to `unsafe_allow_html`.

## UI authentication

The Streamlit UI can spend money and read reports, so it fails closed unless `MARS_UI_PASSWORD` is set. Each signed-in user gets a separate tenant directory under `research_output/`, the cache, dead-letter queue, traces and approvals. `MARS_UI_ALLOW_ANONYMOUS=1` is the explicit, local-demo-only opt-out.

## Tests

```bash
uv run pytest -q                 # hermetic suite (live tests deselected)
uv run pytest -m live -q         # opt-in real-provider contract test
```

The suite covers the evidence store, the HTTP layer, the retrieval tools, the LLM providers, the report renderer, the whole pipeline, and the P1/P2 operational surface: atomic persistence, startup validation, spend budgets, shared breakers and rate limits, cache eviction/TTL, retention, health endpoints, resume/replay, concurrency and fault injection, immutable evidence archives, entailment auditing, crawler governance, prompt-injection sanitisation, redaction, encryption at rest, disclosure/deletion and offline report verification. The release surface has its own tests: the pinned model registry and lifecycle policy, prompt version/channel/override validation, profile-to-contract consistency, per-agent/per-track cost attribution and budgets, and the evaluation dataset, gate cases, report assertions and baseline comparison. The Agentic AI domain lock has its own tests for the topic gate, taxonomy mapping, plan normalisation and retrieval filtering. It is hermetic: no default test makes an outbound network request (the HTTP client refuses to connect, and network tools are withheld). `tests/conftest.py` contains a local server that speaks the OpenAI chat-completions protocol, so the real client stack (`langchain-openrouter` → OpenRouter SDK → HTTP → CrewAI's tool-calling loop) is exercised end to end without a key or network access. The deterministic provider and test corpus live under `tests/support/` and are injected only by the test harness; production has no mock, offline or curated-corpus path. CI runs the suite and the `mars eval` gate cases on every push, so the domain lock and dataset cannot drift unnoticed.

## Layout

```
evals/
└── agentic_ai_v1.json # fixed regression set (gate, injection and research cases)

src/mars/
├── agents/        # CrewAI personas, prompt templates, contracts, prompt registry, crew factory
├── tools/         # search, fetch, papers, findings ledger (governed + sanitised)
├── archive.py     # immutable, content-addressed evidence snapshots
├── config.py      # Settings + environment handling + validation
├── context.py     # shared runtime context (settings, http, evidence, governance)
├── crypto.py      # optional encryption at rest
├── data_governance.py # disclosure and run deletion
├── domain.py      # fixed Agentic AI domain, taxonomy, topic gate, relevance filter
├── entailment.py  # deterministic claim/sentence entailment checks
├── evaluation_set.py # fixed regression dataset, report assertions, baseline comparison
├── evidence.py    # source registry + claim ledger (snapshot/restore)
├── governance.py  # robots.txt, domain, quality, paywall and licence policy
├── health.py      # /healthz, /readyz, /metrics server
├── llm.py         # OpenRouter provider (single production provider)
├── metrics.py     # usage, pricing, spend ceilings, per-agent/track attribution
├── model_registry.py # pinned model ids, lifecycle status and fallback chain
├── net.py         # cached, rate-limited HTTP client
├── observability.py # sampled JSONL traces + structured logging
├── pipeline.py    # orchestration, resume/replay, atomic persistence
├── redaction.py   # secret/PII redaction
├── reliability.py # breakers, budgets, watchdog, DLQ, run-state journal
├── reporting.py   # Markdown report renderer
├── retention.py   # retention and cache pruning
├── sanitize.py    # prompt-injection neutralisation
├── storage.py     # atomic writes and unique run ids
├── verification.py # offline report re-verification
├── cli.py         # Typer CLI
└── app.py         # Streamlit UI (visualizations, history, export, search)
```

## Licence

MIT.
