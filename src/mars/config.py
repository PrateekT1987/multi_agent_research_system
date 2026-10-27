"""Configuration for the Mars multi-agent research system.

Everything is driven by environment variables (optionally loaded from a ``.env``
file) so that the system can run unchanged in a shell, in CI, or inside the
Streamlit UI.  :class:`Settings` is a plain dataclass: it is cheap to build,
easy to override in tests, and easy to serialise into a run manifest.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .domain import (
    DEFAULT_DOMAIN_TRACKS,
    DOMAIN,
    DomainError,
    canonical_track,
    ensure_domain_topic,
)
from .model_registry import (
    DEFAULT_MODEL,
    ModelRegistryError,
    model_pin,
    resolve_model_chain,
    validate_model,
)
from .prompt_meta import PROMPT_CHANNEL, PROMPT_CHANNELS, PROMPT_VERSION

__all__ = [
    "Settings",
    "ConfigError",
    "TRACKS",
    "load_dotenv",
    "build_settings",
]

DEFAULT_DATA_DIR = Path.home() / ".cache" / "mars"


class ConfigError(ValueError):
    """Invalid or unsafe runtime configuration detected before a run starts."""


# Mars has one domain.  The Director chooses a subset of this fixed Agentic AI
# taxonomy; the pipeline never invents general-purpose tracks.
TRACKS = DEFAULT_DOMAIN_TRACKS

REDACTED = "***redacted***"
SECRET_FIELDS = frozenset(
    {
        "openrouter_api_key",
        "brave_api_key",
        "serper_api_key",
        "tavily_api_key",
        "exa_api_key",
        "approval_secret",
        "encryption_key",
    }
)
_SECRET_MARKERS = ("key", "token", "secret", "password", "credential")


def _redact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    """Redact obvious secrets from free-form settings such as ``extra``."""
    return {
        key: (REDACTED if any(marker in str(key).lower() for marker in _SECRET_MARKERS) else value)
        for key, value in values.items()
    }


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Minimal ``.env`` loader.

    We avoid a dependency on ``python-dotenv`` for two reasons: it is not
    guaranteed to be installed, and we only need a tiny subset of its
    behaviour (``KEY=value`` lines, ``#`` comments, optional ``export``).
    """
    env_path = Path(path).expanduser()
    loaded: dict[str, str] = {}
    if not env_path.is_file():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _env_secret(name: str) -> str | None:
    """Read a secret from ``<NAME>_FILE`` (Docker/K8s secret) or the env.

    Mounted secret files are the portable way to integrate with a secret
    manager without teaching Mars about Vault/AWS/GCP SDKs.
    """
    path = os.environ.get(f"{name}_FILE")
    if path:
        try:
            return Path(path).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return os.environ.get(name)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int_optional(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_float_optional(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(parts) or default


@dataclass(slots=True)
class Settings:
    """Runtime configuration for a research run."""

    # ------------------------------------------------------------------ topic
    # The domain is fixed; ``topic`` selects a subtopic inside Agentic AI.
    topic: str = "agentic AI"
    tracks: tuple[str, ...] = TRACKS
    depth: str = "standard"  # quick | standard | deep
    language: str = "en"

    # ------------------------------------------------------------------- llm
    llm_provider: str = "openrouter"
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_strong: str = DEFAULT_MODEL
    model_fast: str = DEFAULT_MODEL
    model_planner: str = DEFAULT_MODEL
    model_fallbacks: tuple[str, ...] = ()
    strict_model_registry: bool = False
    temperature: float = 0.25
    max_tokens: int = 8192
    request_timeout: float = 120.0
    max_llm_retries: int = 2
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 20.0
    circuit_failure_threshold: int = 4
    circuit_recovery_seconds: float = 30.0
    circuit_wait_seconds: float = 40.0
    max_llm_calls_per_run: int = 160
    max_tokens_per_run: int | None = None
    max_cost_usd: float | None = None
    max_tokens_per_track: int | None = None
    max_cost_usd_per_track: float | None = None
    strict_pricing: bool = False
    observability_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR / "traces")
    trace_sample_rate: float = 1.0
    otel_endpoint: str | None = None
    alert_webhook_url: str | None = None
    enable_continuous_evaluation: bool = False
    prompt_version: str = PROMPT_VERSION
    prompt_channel: str = PROMPT_CHANNEL
    prompt_overrides_path: str | None = None
    max_run_seconds: float = 1800.0
    max_track_seconds: float = 480.0
    shutdown_grace_seconds: float = 10.0

    # -------------------------------------------------------------- egress
    # Set ``allow_network=False`` to refuse every outbound request and withhold
    # network tools.  Even when network access is enabled, private, loopback and
    # link-local addresses are refused unless ``allow_private_network`` is
    # explicitly turned on for local dev.
    allow_network: bool = True
    allow_private_network: bool = False
    max_response_bytes: int = 10_000_000
    http_max_redirects: int = 5

    # ----------------------------------------------------------------- search
    search_providers: tuple[str, ...] = ("tavily", "brave", "serper", "exa")
    brave_api_key: str | None = None
    serper_api_key: str | None = None
    tavily_api_key: str | None = None
    exa_api_key: str | None = None
    max_search_results: int = 8
    max_page_chars: int = 12000

    # -------------------------------------------------------------- governance
    respect_robots: bool = True
    robots_fail_closed: bool = False
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    blocked_url_patterns: tuple[str, ...] = ()
    min_source_quality: float = 0.35
    max_sources_per_domain: int = 5
    max_source_age_days: int | None = None
    block_paywalled: bool = False

    # ------------------------------------------------------------------- net
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 MarsResearchBot/0.1 (+research agent)"
    )
    http_timeout: float = 25.0
    http_max_retries: int = 3
    requests_per_minute: int = 40
    min_seconds_between_requests: float = 0.6

    # ---------------------------------------------------------------- cache
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    cache_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR / "cache")
    cache_max_bytes: int = 500_000_000
    cache_max_entries: int = 50_000
    llm_cache_ttl_hours: int = 168
    llm_cache_max_entries: int = 10_000
    run_state_dir: Path | None = None

    # ------------------------------------------------------- evidence / privacy
    archive_enabled: bool = True
    archive_dir: Path | None = None
    archive_max_chars: int = 2_000_000
    redact_pii: bool = True
    # Optional Fernet key.  When set, state, dead letters and approval payloads
    # are encrypted at rest.  Accepts a Fernet key or any high-entropy string.
    encryption_key: str | None = None
    encrypt_state: bool = False
    entailment_supported_at: float = 0.55
    entailment_partial_at: float = 0.30
    min_entailment_ratio: float = 0.55

    # ------------------------------------------------------------- pipeline
    max_rounds: int = 2  # critique -> research -> rewrite cycles
    max_tool_iterations: int = 14
    max_parallel_tracks: int = 3
    max_tracks: int = 6
    grounding_threshold: float = 0.85
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "research_output")
    dead_letter_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR / "dead_letters")
    retention_days: int = 30
    ui_poll_seconds: float = 1.0
    verbose: bool = True
    crew_verbose: bool = False  # CrewAI's own (very loud) console output

    # ------------------------------------------------------------ human-in-loop
    require_human_approval: bool = False
    approval_checkpoints: tuple[str, ...] = ("plan", "draft")
    # Human gates must never wait forever.  Operators can raise or lower this
    # with ``MARS_APPROVAL_TIMEOUT_SECONDS``; ``0`` fails immediately.
    approval_timeout_seconds: int | None = 1800
    approval_auto_approve: bool = False
    require_human_on_degraded: bool = False
    approval_dir: Path = field(default_factory=lambda: Path.cwd() / ".mars_approvals")
    # When set, approval/rejection files must carry a valid HMAC signature.
    approval_secret: str | None = None

    # --------------------------------------------------------------- runtime
    run_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- utilities
    def with_overrides(self, **kwargs: Any) -> Settings:
        """Return a copy with the given non-``None`` fields replaced."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)

    @property
    def has_live_llm(self) -> bool:
        return self.llm_provider == "openrouter" and bool(self.openrouter_api_key)

    @property
    def data_dir(self) -> Path:
        return self.cache_dir.parent

    @property
    def state_dir(self) -> Path:
        return self.run_state_dir or (self.data_dir / "runs")

    @property
    def evidence_dir(self) -> Path:
        return self.archive_dir or (self.data_dir / "evidence")

    def warnings(self) -> list[str]:
        """Non-fatal configuration problems worth surfacing at startup."""
        from .metrics import pricing_known

        messages: list[str] = []
        for role, model in (
            ("model_strong", self.model_strong),
            ("model_fast", self.model_fast),
            ("model_planner", self.model_planner),
        ):
            if not pricing_known(model):
                messages.append(
                    f"model {model!r} has no pricing entry; cost estimates will be marked unknown"
                )
            try:
                messages.extend(validate_model(model, role=role))
            except ModelRegistryError:
                # A retired primary is a hard error in validate() unless an
                # active fallback is configured; surface the substitution here.
                try:
                    chain = resolve_model_chain(model, self.model_fallbacks)
                except ModelRegistryError:
                    pass
                else:
                    if chain[0] != model:
                        messages.append(
                            f"{role} {model!r} is retired; the run will use "
                            f"fallback {chain[0]!r}"
                        )
        if self.prompt_channel != "prod":
            messages.append(
                f"prompt channel is {self.prompt_channel!r}; production runs should use 'prod'"
            )
        if not self.search_providers:
            messages.append("no search providers configured; web_search will return nothing")
        return messages

    def validate(self) -> None:
        """Fail fast on unsafe or nonsensical configuration.

        This is called by the pipeline and CLI, not by :func:`build_settings`,
        so library callers can still construct partial settings for tests.
        """
        from .metrics import pricing_known

        errors: list[str] = []
        if self.depth not in {"quick", "standard", "deep"}:
            errors.append(f"unknown depth {self.depth!r} (expected quick, standard or deep)")
        if self.llm_provider != "openrouter":
            errors.append(
                f"unknown llm_provider {self.llm_provider!r}; only 'openrouter' is supported"
            )
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is required for the openrouter provider")

        # Model pinning: a retired model must never be used silently.  A retired
        # primary is allowed only when an active fallback exists, in which case
        # the pipeline resolves to that fallback before the first call.  An
        # unpinned model is allowed by default (custom OpenRouter routes) but
        # becomes an error when strict_model_registry is enabled.
        for role, model in (
            ("model_strong", self.model_strong),
            ("model_fast", self.model_fast),
            ("model_planner", self.model_planner),
        ):
            try:
                validate_model(model, role=role)
            except ModelRegistryError as exc:
                pin = model_pin(model)
                if pin is not None and pin.status == "retired":
                    try:
                        chain = resolve_model_chain(model, self.model_fallbacks)
                    except ModelRegistryError:
                        errors.append(str(exc))
                    else:
                        if chain[0] == model:
                            errors.append(str(exc))
                else:
                    errors.append(str(exc))
            if self.strict_model_registry and model_pin(model) is None:
                errors.append(
                    f"{role} {model!r} is not pinned in the model registry "
                    "(strict_model_registry is enabled)"
                )
        for model in self.model_fallbacks:
            pin = model_pin(model)
            if pin is not None and pin.status == "retired":
                # Retired fallbacks are dropped by resolve_model_chain rather
                # than failing the whole chain.
                continue
            try:
                validate_model(model, role="fallback model")
            except ModelRegistryError as exc:
                errors.append(str(exc))
            if self.strict_model_registry and pin is None:
                errors.append(
                    f"fallback model {model!r} is not pinned in the model registry "
                    "(strict_model_registry is enabled)"
                )

        # Prompt release: the selected version/channel and any override file
        # must be internally consistent before a run starts.
        if self.prompt_channel not in PROMPT_CHANNELS:
            errors.append(
                f"unknown prompt channel {self.prompt_channel!r}; "
                f"expected one of {', '.join(PROMPT_CHANNELS)}"
            )
        try:
            from .agents.prompt_registry import PromptRegistryError, load_prompt_bundle

            load_prompt_bundle(
                self.prompt_overrides_path,
                version=self.prompt_version,
                channel=self.prompt_channel,
            )
        except PromptRegistryError as exc:
            errors.append(str(exc))
        except Exception as exc:  # pragma: no cover - defensive file handling
            errors.append(f"invalid prompt configuration: {exc}")

        # Mars is domain-locked.  Reject an off-domain topic before any model
        # call or network request, and keep the track fallback inside the
        # canonical Agentic AI taxonomy.
        try:
            ensure_domain_topic(self.topic)
        except DomainError as exc:
            errors.append(str(exc))
        for track in self.tracks:
            if canonical_track(track) is None:
                errors.append(
                    f"unknown track {track!r}; tracks must belong to the "
                    f"{DOMAIN} taxonomy"
                )

        for name, value, minimum in (
            ("max_rounds", self.max_rounds, 1),
            ("max_tool_iterations", self.max_tool_iterations, 1),
            ("max_parallel_tracks", self.max_parallel_tracks, 1),
            ("max_tracks", self.max_tracks, 1),
            ("max_llm_calls_per_run", self.max_llm_calls_per_run, 1),
            ("max_search_results", self.max_search_results, 1),
            ("max_page_chars", self.max_page_chars, 500),
            ("max_response_bytes", self.max_response_bytes, 1),
            ("cache_max_bytes", self.cache_max_bytes, 1),
            ("cache_max_entries", self.cache_max_entries, 1),
            ("llm_cache_max_entries", self.llm_cache_max_entries, 1),
            ("retention_days", self.retention_days, 0),
        ):
            if value < minimum:
                errors.append(f"{name} must be >= {minimum} (got {value})")

        for name, value in (
            ("http_timeout", self.http_timeout),
            ("request_timeout", self.request_timeout),
            ("shutdown_grace_seconds", self.shutdown_grace_seconds),
            ("ui_poll_seconds", self.ui_poll_seconds),
        ):
            if value <= 0:
                errors.append(f"{name} must be > 0 (got {value})")

        if self.http_max_redirects < 0:
            errors.append("http_max_redirects must be >= 0")
        if self.circuit_wait_seconds < 0:
            errors.append("circuit_wait_seconds must be >= 0")
        if self.approval_timeout_seconds is not None and self.approval_timeout_seconds < 0:
            errors.append("approval_timeout_seconds must be >= 0")
        if not 0.0 <= self.trace_sample_rate <= 1.0:
            errors.append("trace_sample_rate must be between 0 and 1")
        if self.max_tokens_per_run is not None and self.max_tokens_per_run <= 0:
            errors.append("max_tokens_per_run must be > 0 when set")
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            errors.append("max_cost_usd must be >= 0 when set")
        if self.max_tokens_per_track is not None and self.max_tokens_per_track <= 0:
            errors.append("max_tokens_per_track must be > 0 when set")
        if self.max_cost_usd_per_track is not None and self.max_cost_usd_per_track < 0:
            errors.append("max_cost_usd_per_track must be >= 0 when set")
        if self.max_source_age_days is not None and self.max_source_age_days < 0:
            errors.append("max_source_age_days must be >= 0 when set")
        if not 0.0 <= self.min_source_quality <= 1.0:
            errors.append("min_source_quality must be between 0 and 1")
        if self.max_sources_per_domain < 0:
            errors.append("max_sources_per_domain must be >= 0 (0 disables the cap)")
        if not 0.0 <= self.entailment_partial_at <= self.entailment_supported_at <= 1.0:
            errors.append(
                "entailment thresholds must satisfy 0 <= partial_at <= supported_at <= 1"
            )
        if not 0.0 <= self.min_entailment_ratio <= 1.0:
            errors.append("min_entailment_ratio must be between 0 and 1")
        if self.encrypt_state and not self.encryption_key:
            errors.append("MARS_ENCRYPTION_KEY is required when encrypt_state is enabled")
        overlap = set(self.allowed_domains) & set(self.blocked_domains)
        if overlap:
            errors.append(
                f"domains cannot be both allowed and blocked: {', '.join(sorted(overlap))}"
            )

        for name in ("openrouter_base_url",):
            value = getattr(self, name)
            parts = urlsplit(value or "")
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                errors.append(f"{name} must be an absolute http(s) URL (got {value!r})")

        if self.alert_webhook_url:
            parts = urlsplit(self.alert_webhook_url)
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                errors.append("alert_webhook_url must be an absolute http(s) URL")

        valid_providers = {"duckduckgo", "brave", "serper", "tavily", "exa"}
        unknown = sorted(set(self.search_providers) - valid_providers)
        if unknown:
            errors.append(f"unknown search provider(s): {', '.join(unknown)}")

        if self.strict_pricing:
            for model in {self.model_strong, self.model_fast, self.model_planner}:
                if not pricing_known(model):
                    errors.append(f"strict pricing is enabled but {model!r} has no price")

        if self.require_human_approval and not self.approval_secret:
            errors.append("MARS_APPROVAL_SECRET is required when human approval is enabled")

        if errors:
            raise ConfigError("; ".join(errors))

    def to_dict(self, *, redact_secrets: bool = True) -> dict[str, Any]:
        """Serialise settings for diagnostics and run manifests.

        Secrets are redacted by default: run manifests are routinely copied,
        archived and attached to bug reports, so a plain ``asdict`` would leak
        provider keys.  Pass ``redact_secrets=False`` only for an in-memory
        consumer that genuinely needs the raw values.
        """
        data = asdict(self)
        data["domain"] = DOMAIN
        data["cache_dir"] = str(self.cache_dir)
        data["output_dir"] = str(self.output_dir)
        data["dead_letter_dir"] = str(self.dead_letter_dir)
        data["observability_dir"] = str(self.observability_dir)
        data["approval_dir"] = str(self.approval_dir)
        data["archive_dir"] = str(self.evidence_dir) if self.archive_enabled else None
        # ``run_state_dir`` is ``None`` unless MARS_STATE_DIR overrides the
        # derived location; when it is set it must still serialise as text so
        # manifests and metrics can be JSON-encoded.
        data["run_state_dir"] = str(self.run_state_dir) if self.run_state_dir else None
        if redact_secrets:
            for name in SECRET_FIELDS:
                if data.get(name):
                    data[name] = REDACTED
            if isinstance(data.get("extra"), dict):
                data["extra"] = _redact_mapping(data["extra"])
        return data


def _default(name: str) -> Any:
    """Return the declared default of a :class:`Settings` field.

    ``Settings`` is a ``slots=True`` dataclass, so reading ``Settings.topic``
    on the *class* returns the slot descriptor rather than the default value.
    Reading it from ``__dataclass_fields__`` is the only reliable way.
    """
    field = Settings.__dataclass_fields__[name]
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return field.default_factory()  # type: ignore[misc]
    return None


def build_settings(
    *,
    env_file: str | Path | None = ".env",
    topic: str | None = None,
    **overrides: Any,
) -> Settings:
    """Build :class:`Settings` from the environment, then apply overrides."""
    if env_file:
        load_dotenv(env_file)

    settings = Settings(
        topic=topic or os.environ.get("MARS_TOPIC", _default("topic")),
        llm_provider=os.environ.get("MARS_LLM_PROVIDER", _default("llm_provider")),
        openrouter_api_key=_env_secret("OPENROUTER_API_KEY"),
        openrouter_base_url=os.environ.get(
            "OPENROUTER_BASE_URL", _default("openrouter_base_url")
        ),
        model_strong=os.environ.get("MARS_MODEL_STRONG", _default("model_strong")),
        model_fast=os.environ.get("MARS_MODEL_FAST", _default("model_fast")),
        model_planner=os.environ.get("MARS_MODEL_PLANNER", _default("model_planner")),
        model_fallbacks=_env_list("MARS_MODEL_FALLBACKS", _default("model_fallbacks")),
        strict_model_registry=_env_bool(
            "MARS_STRICT_MODEL_REGISTRY", _default("strict_model_registry")
        ),
        max_llm_retries=_env_int("MARS_MAX_LLM_RETRIES", _default("max_llm_retries")),
        retry_base_seconds=_env_float("MARS_RETRY_BASE_SECONDS", _default("retry_base_seconds")),
        retry_max_seconds=_env_float("MARS_RETRY_MAX_SECONDS", _default("retry_max_seconds")),
        circuit_failure_threshold=_env_int(
            "MARS_CIRCUIT_FAILURE_THRESHOLD", _default("circuit_failure_threshold")
        ),
        circuit_recovery_seconds=_env_float(
            "MARS_CIRCUIT_RECOVERY_SECONDS", _default("circuit_recovery_seconds")
        ),
        circuit_wait_seconds=_env_float(
            "MARS_CIRCUIT_WAIT_SECONDS", _default("circuit_wait_seconds")
        ),
        max_llm_calls_per_run=_env_int("MARS_MAX_LLM_CALLS_PER_RUN", _default("max_llm_calls_per_run")),
        max_tokens_per_run=_env_int_optional("MARS_MAX_TOKENS_PER_RUN"),
        max_cost_usd=_env_float_optional("MARS_MAX_COST_USD"),
        max_tokens_per_track=_env_int_optional("MARS_MAX_TOKENS_PER_TRACK"),
        max_cost_usd_per_track=_env_float_optional("MARS_MAX_COST_USD_PER_TRACK"),
        strict_pricing=_env_bool("MARS_STRICT_PRICING", _default("strict_pricing")),
        trace_sample_rate=_env_float("MARS_TRACE_SAMPLE_RATE", _default("trace_sample_rate")),
        otel_endpoint=os.environ.get("MARS_OTEL_ENDPOINT") or None,
        alert_webhook_url=os.environ.get("MARS_ALERT_WEBHOOK_URL") or None,
        shutdown_grace_seconds=_env_float(
            "MARS_SHUTDOWN_GRACE_SECONDS", _default("shutdown_grace_seconds")
        ),
        max_run_seconds=_env_float("MARS_MAX_RUN_SECONDS", _default("max_run_seconds")),
        max_track_seconds=_env_float("MARS_MAX_TRACK_SECONDS", _default("max_track_seconds")),
        max_tracks=_env_int("MARS_MAX_TRACKS", _default("max_tracks")),
        allow_network=_env_bool("MARS_ALLOW_NETWORK", _default("allow_network")),
        allow_private_network=_env_bool(
            "MARS_ALLOW_PRIVATE_NETWORK", _default("allow_private_network")
        ),
        max_response_bytes=_env_int("MARS_MAX_RESPONSE_BYTES", _default("max_response_bytes")),
        http_max_redirects=_env_int("MARS_HTTP_MAX_REDIRECTS", _default("http_max_redirects")),
        search_providers=_env_list(
            "MARS_SEARCH_PROVIDERS", _default("search_providers")
        ),
        brave_api_key=_env_secret("BRAVE_API_KEY"),
        serper_api_key=_env_secret("SERPER_API_KEY"),
        tavily_api_key=_env_secret("TAVILY_API_KEY"),
        exa_api_key=_env_secret("EXA_API_KEY"),
        respect_robots=_env_bool("MARS_RESPECT_ROBOTS", _default("respect_robots")),
        robots_fail_closed=_env_bool("MARS_ROBOTS_FAIL_CLOSED", _default("robots_fail_closed")),
        allowed_domains=_env_list("MARS_ALLOWED_DOMAINS", _default("allowed_domains")),
        blocked_domains=_env_list("MARS_BLOCKED_DOMAINS", _default("blocked_domains")),
        blocked_url_patterns=_env_list(
            "MARS_BLOCKED_URL_PATTERNS", _default("blocked_url_patterns")
        ),
        min_source_quality=_env_float("MARS_MIN_SOURCE_QUALITY", _default("min_source_quality")),
        max_sources_per_domain=_env_int(
            "MARS_MAX_SOURCES_PER_DOMAIN", _default("max_sources_per_domain")
        ),
        max_source_age_days=_env_int_optional("MARS_MAX_SOURCE_AGE_DAYS"),
        block_paywalled=_env_bool("MARS_BLOCK_PAYWALLED", _default("block_paywalled")),
        archive_enabled=_env_bool("MARS_ARCHIVE_ENABLED", _default("archive_enabled")),
        archive_dir=(
            Path(os.environ["MARS_ARCHIVE_DIR"]) if os.environ.get("MARS_ARCHIVE_DIR") else None
        ),
        archive_max_chars=_env_int("MARS_ARCHIVE_MAX_CHARS", _default("archive_max_chars")),
        redact_pii=_env_bool("MARS_REDACT_PII", _default("redact_pii")),
        encryption_key=_env_secret("MARS_ENCRYPTION_KEY") or None,
        encrypt_state=_env_bool("MARS_ENCRYPT_STATE", _default("encrypt_state")),
        entailment_supported_at=_env_float(
            "MARS_ENTAILMENT_SUPPORTED_AT", _default("entailment_supported_at")
        ),
        entailment_partial_at=_env_float(
            "MARS_ENTAILMENT_PARTIAL_AT", _default("entailment_partial_at")
        ),
        min_entailment_ratio=_env_float(
            "MARS_MIN_ENTAILMENT_RATIO", _default("min_entailment_ratio")
        ),
        cache_dir=Path(os.environ.get("MARS_CACHE_DIR", str(DEFAULT_DATA_DIR / "cache"))),
        cache_max_bytes=_env_int("MARS_CACHE_MAX_BYTES", _default("cache_max_bytes")),
        cache_max_entries=_env_int("MARS_CACHE_MAX_ENTRIES", _default("cache_max_entries")),
        llm_cache_ttl_hours=_env_int(
            "MARS_LLM_CACHE_TTL_HOURS", _default("llm_cache_ttl_hours")
        ),
        llm_cache_max_entries=_env_int(
            "MARS_LLM_CACHE_MAX_ENTRIES", _default("llm_cache_max_entries")
        ),
        run_state_dir=(
            Path(os.environ["MARS_STATE_DIR"])
            if os.environ.get("MARS_STATE_DIR")
            else None
        ),
        output_dir=Path(os.environ.get("MARS_OUTPUT_DIR", str(Path.cwd() / "research_output"))),
        dead_letter_dir=Path(os.environ.get("MARS_DEAD_LETTER_DIR", str(DEFAULT_DATA_DIR / "dead_letters"))),
        retention_days=_env_int("MARS_RETENTION_DAYS", _default("retention_days")),
        ui_poll_seconds=_env_float("MARS_UI_POLL_SECONDS", _default("ui_poll_seconds")),
        observability_dir=Path(os.environ.get("MARS_OBSERVABILITY_DIR", str(DEFAULT_DATA_DIR / "traces"))),
        enable_continuous_evaluation=_env_bool(
            "MARS_ENABLE_CONTINUOUS_EVALUATION", _default("enable_continuous_evaluation")
        ),
        prompt_version=os.environ.get("MARS_PROMPT_VERSION", _default("prompt_version")),
        prompt_channel=os.environ.get("MARS_PROMPT_CHANNEL", _default("prompt_channel")),
        prompt_overrides_path=os.environ.get("MARS_PROMPT_OVERRIDES") or None,
        depth=os.environ.get("MARS_DEPTH", _default("depth")),
        verbose=_env_bool("MARS_VERBOSE", _default("verbose")),
        crew_verbose=_env_bool("MARS_CREW_VERBOSE", _default("crew_verbose")),
        require_human_approval=_env_bool("MARS_REQUIRE_HUMAN_APPROVAL", _default("require_human_approval")),
        approval_checkpoints=_env_list(
            "MARS_APPROVAL_CHECKPOINTS", _default("approval_checkpoints")
        ),
        approval_auto_approve=_env_bool("MARS_APPROVAL_AUTO_APPROVE", _default("approval_auto_approve")),
        require_human_on_degraded=_env_bool(
            "MARS_REQUIRE_HUMAN_ON_DEGRADED", _default("require_human_on_degraded")
        ),
        approval_timeout_seconds=_env_int(
            "MARS_APPROVAL_TIMEOUT_SECONDS", _default("approval_timeout_seconds")
        ),
        approval_dir=Path(
            os.environ.get("MARS_APPROVAL_DIR", str(Path.cwd() / ".mars_approvals"))
        ),
        approval_secret=_env_secret("MARS_APPROVAL_SECRET") or None,
    )

    settings = settings.with_overrides(**overrides)
    if settings.depth == "quick":
        settings = settings.with_overrides(
            max_rounds=1,
            max_search_results=5,
            max_tool_iterations=8,
        )
    elif settings.depth == "deep":
        settings = settings.with_overrides(
            max_rounds=3,
            max_search_results=12,
            max_tool_iterations=20,
        )
    return settings
