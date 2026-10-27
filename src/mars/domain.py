"""The Agentic AI domain lock.

Mars is not a general-purpose research system.  Its only domain is **Agentic
AI**: AI systems that plan, reason, use tools, remember context and act,
including single-agent loops and multi-agent orchestration.

The lock is enforced at four layers:

* :func:`ensure_domain_topic` rejects an off-domain topic before any model call,
* the Director may only choose tracks from :data:`DOMAIN_TRACKS`,
* every agent prompt carries :func:`domain_protocol`, and
* retrieval results must pass :func:`is_domain_relevant`.

Keeping the vocabulary and the taxonomy in one module makes the domain easy to
audit and impossible to widen by accident from a prompt or an environment
variable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "DOMAIN",
    "DOMAIN_SLUG",
    "DOMAIN_DESCRIPTION",
    "DOMAIN_QUERY_SUFFIX",
    "DOMAIN_TRACKS",
    "DOMAIN_TRACK_KEYS",
    "DEFAULT_DOMAIN_TRACKS",
    "SUBTOPIC_EXAMPLES",
    "DomainError",
    "DomainTrack",
    "canonical_track",
    "domain_protocol",
    "ensure_domain_topic",
    "is_domain_relevant",
    "is_domain_topic",
    "scope_query",
    "track_menu",
    "track_title",
]

DOMAIN = "Agentic AI"
DOMAIN_SLUG = "agentic-ai"
DOMAIN_DESCRIPTION = (
    "AI systems that plan, reason, use tools, remember context and act - "
    "from a single LLM agent loop to multi-agent orchestration."
)
DOMAIN_QUERY_SUFFIX = "agentic AI"

SUBTOPIC_EXAMPLES = (
    "agent memory",
    "multi-agent orchestration",
    "tool use and function calling",
    "computer-use agents",
    "agent evaluation and benchmarks",
    "agent safety and prompt injection",
    "agent frameworks and protocols",
    "the economics of agent deployment",
)


class DomainError(ValueError):
    """Raised when a request falls outside the Agentic AI domain."""


@dataclass(frozen=True, slots=True)
class DomainTrack:
    """One canonical track in the Agentic AI research taxonomy."""

    key: str
    title: str
    objective: str


DOMAIN_TRACKS: tuple[DomainTrack, ...] = (
    DomainTrack(
        key="foundations",
        title="Foundations and Origins",
        objective=(
            "Trace the ideas, systems and papers that established goal-directed, "
            "tool-using AI agents, from symbolic agents and reinforcement learning "
            "to ReAct and the modern LLM agent loop."
        ),
    ),
    DomainTrack(
        key="architectures",
        title="Agent Architectures and Orchestration",
        objective=(
            "Explain how agentic AI is built: the agent loop, planning, memory, "
            "tool use, function calling and multi-agent orchestration."
        ),
    ),
    DomainTrack(
        key="capabilities",
        title="Capabilities, Benchmarks and Evaluation",
        objective=(
            "Establish what agentic AI systems can do, how they are evaluated and "
            "benchmarked, and where they still fail."
        ),
    ),
    DomainTrack(
        key="applications",
        title="Applications and Deployment",
        objective=(
            "Document where agentic AI is deployed, which workflows it changes and "
            "what the evidence says about its impact."
        ),
    ),
    DomainTrack(
        key="safety",
        title="Safety, Security and Governance",
        objective=(
            "Cover prompt injection, security, reliability, human oversight, "
            "regulation and responsible deployment of agentic AI."
        ),
    ),
    DomainTrack(
        key="ecosystem",
        title="Ecosystem, Adoption and Economics",
        objective=(
            "Map the agentic AI ecosystem: frameworks, protocols such as MCP and "
            "A2A, vendors, adoption and deployment economics."
        ),
    ),
    DomainTrack(
        key="future",
        title="Future Trajectory",
        objective=(
            "Identify credible near-term trajectories for agentic AI, the open "
            "problems and the signals worth watching."
        ),
    ),
)

DOMAIN_TRACK_KEYS: tuple[str, ...] = tuple(track.key for track in DOMAIN_TRACKS)

# The fallback plan used when the Director cannot produce a usable plan.
DEFAULT_DOMAIN_TRACKS: tuple[str, ...] = (
    "foundations",
    "architectures",
    "capabilities",
    "applications",
    "safety",
    "future",
)

_BY_KEY = {track.key: track for track in DOMAIN_TRACKS}

# Aliases let a Director (or an older persisted plan) name a track loosely while
# still landing inside the fixed taxonomy.
_TRACK_ALIASES: dict[str, str] = {
    # foundations
    "history": "foundations",
    "historical": "foundations",
    "origins": "foundations",
    "background": "foundations",
    "precursors": "foundations",
    "evolution": "foundations",
    "early": "foundations",
    "timeline": "foundations",
    # architectures
    "architecture": "architectures",
    "agent_architecture": "architectures",
    "orchestration": "architectures",
    "multi_agent": "architectures",
    "multi_agent_systems": "architectures",
    "multiagent": "architectures",
    "tool_use": "architectures",
    "tooling": "architectures",
    "planning": "architectures",
    "memory": "architectures",
    "agent_loop": "architectures",
    # capabilities
    "current": "capabilities",
    "current_landscape": "capabilities",
    "state_of_art": "capabilities",
    "state_of_the_art": "capabilities",
    "present": "capabilities",
    "today": "capabilities",
    "capability": "capabilities",
    "benchmark": "capabilities",
    "benchmarks": "capabilities",
    "evaluation": "capabilities",
    "evals": "capabilities",
    "reliability": "capabilities",
    "performance": "capabilities",
    # applications
    "application": "applications",
    "use_cases": "applications",
    "deployment": "applications",
    "deployments": "applications",
    "case_studies": "applications",
    "enterprise": "applications",
    # safety
    "ethics": "safety",
    "security": "safety",
    "governance": "safety",
    "policy": "safety",
    "regulation": "safety",
    "risk": "safety",
    "risks": "safety",
    "guardrails": "safety",
    "alignment": "safety",
    "oversight": "safety",
    # ecosystem
    "adoption": "ecosystem",
    "economics": "ecosystem",
    "market": "ecosystem",
    "frameworks": "ecosystem",
    "protocols": "ecosystem",
    "vendors": "ecosystem",
    "commercial": "ecosystem",
    # future
    "trajectory": "future",
    "outlook": "future",
    "forecast": "future",
    "forecasts": "future",
    "roadmap": "future",
}

# ---------------------------------------------------------------------------
# Topic gate
# ---------------------------------------------------------------------------
_STRONG_PHRASES = (
    "agentic",
    "agentic ai",
    "agentic system",
    "agentic systems",
    "agentic workflow",
    "agentic workflows",
    "ai agent",
    "ai agents",
    "llm agent",
    "llm agents",
    "language model agent",
    "language model agents",
    "autonomous agent",
    "autonomous agents",
    "multi agent",
    "multi agents",
    "multiagent",
    "multiagent systems",
    "agent architecture",
    "agent architectures",
    "agent orchestration",
    "agent framework",
    "agent frameworks",
    "agent loop",
    "agentic loop",
    "agent memory",
    "agent planning",
    "agent evaluation",
    "agent benchmark",
    "agent benchmarks",
    "agent safety",
    "agent security",
    "agent governance",
    "agent oversight",
    "agent deployment",
    "agent protocol",
    "agent protocols",
    "agent workflow",
    "agent workflows",
    "tool use",
    "tool using",
    "tool calling",
    "function calling",
    "computer use",
    "computer using",
    "browser agent",
    "browser agents",
    "browser use",
    "coding agent",
    "coding agents",
    "software engineering agent",
    "research agent",
    "autonomous workflow",
    "autonomous workflows",
    "model context protocol",
    "agent to agent",
    "agent2agent",
    "reinforcement learning agent",
    "reinforcement learning agents",
    "rl agent",
    "rl agents",
    "intelligent agent",
    "intelligent agents",
    "software agent",
    "software agents",
    "goal directed agent",
    "goal directed agents",
    "react agent",
    "react prompting",
    "react framework",
    "react pattern",
)

# Unambiguous names from the Agentic AI literature and tooling.
_NAMED_PHRASES = (
    "autogpt",
    "babyagi",
    "agentgpt",
    "metagpt",
    "chatdev",
    "langchain",
    "langgraph",
    "llamaindex",
    "crewai",
    "autogen",
    "autogen studio",
    "semantic kernel",
    "openai swarm",
    "agents sdk",
    "openai operator",
    "claude computer use",
    "anthropic computer use",
    "swe agent",
    "openhands",
    "swe bench",
    "webarena",
    "agentbench",
    "tau bench",
    "osworld",
    "generative agents",
    "reflexion",
    "mcp",
    "a2a",
)

# These names are only evidence of agentic AI when they are the whole topic or
# appear alongside a strong anchor; on their own they are ordinary words.
_AMBIGUOUS_TOPICS = frozenset({"react", "devin", "voyager", "gaia", "swarm", "operator", "camel"})

# Phrases that are decisively outside the domain unless a strong Agentic AI
# anchor is also present.
_HARD_OFF_DOMAIN = (
    "travel agent",
    "travel agents",
    "real estate agent",
    "real estate agents",
    "estate agent",
    "estate agents",
    "letting agent",
    "insurance agent",
    "insurance agents",
    "booking agent",
    "talent agent",
    "sports agent",
    "player agent",
    "secret agent",
    "secret agents",
    "double agent",
    "free agent",
    "free agents",
    "literary agent",
    "press agent",
    "customer service agent",
    "customer service agents",
    "call center agent",
    "call centre agent",
    "support agent",
    "support agents",
    "chemical agent",
    "chemical agents",
    "nerve agent",
    "nerve agents",
    "biological agent",
    "biological agents",
    "infectious agent",
    "infectious agents",
    "pathogenic agent",
    "oxidizing agent",
    "reducing agent",
    "catalytic agent",
    "cleaning agent",
    "bleaching agent",
    "emulsifying agent",
    "thickening agent",
    "flavoring agent",
    "flavouring agent",
    "drying agent",
    "wetting agent",
    "contrast agent",
    "therapeutic agent",
    "antimicrobial agent",
    "cytotoxic agent",
    "alkylating agent",
    "agent orange",
)

_AI_COOCCURRENCE = frozenset(
    {
        "ai",
        "llm",
        "llms",
        "autonomous",
        "tool",
        "tools",
        "planning",
        "reasoning",
        "orchestration",
        "memory",
        "prompt",
        "prompts",
        "workflow",
        "workflows",
        "multi",
        "language",
        "model",
        "models",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#_-]*")


def _normalise(text: str) -> str:
    return " ".join((text or "").lower().replace("_", " ").replace("-", " ").split())


def _has_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def _has_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_has_phrase(text, phrase) for phrase in phrases)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text))


def _has_strong_anchor(text: str) -> bool:
    return _has_any_phrase(text, _STRONG_PHRASES) or _has_any_phrase(text, _NAMED_PHRASES)


def _has_agent_cooccurrence(text: str) -> bool:
    tokens = _tokens(text)
    return bool(tokens & {"agent", "agents"}) and bool(tokens & _AI_COOCCURRENCE)


def _is_exact_ambiguous(text: str) -> bool:
    return text.strip() in _AMBIGUOUS_TOPICS


def _off_domain_message(topic: str) -> str:
    examples = ", ".join(SUBTOPIC_EXAMPLES[:5])
    return (
        f"Mars researches only {DOMAIN}. {topic!r} is outside that domain. "
        f"Choose an Agentic AI subtopic such as: {examples}."
    )


def is_domain_topic(topic: str) -> bool:
    """Return ``True`` when ``topic`` is clearly within Agentic AI."""
    text = _normalise(topic)
    if not text:
        return True
    if _has_strong_anchor(text):
        return True
    if _has_agent_cooccurrence(text):
        return True
    if _has_any_phrase(text, _HARD_OFF_DOMAIN):
        return False
    return _is_exact_ambiguous(text)


def ensure_domain_topic(topic: str) -> str:
    """Validate and normalise a topic, defaulting an empty value to the domain.

    Raises :class:`DomainError` when the topic is outside Agentic AI.  Callers
    must treat this as a fail-closed check, not a suggestion.
    """
    cleaned = " ".join((topic or "").split())
    if not cleaned:
        return DOMAIN
    if not is_domain_topic(cleaned):
        raise DomainError(_off_domain_message(cleaned))
    return cleaned


def is_domain_relevant(text: str) -> bool:
    """Whether a snippet, page or abstract carries Agentic AI signal.

    The check is deliberately generous about vocabulary and strict about the
    subject: an ``agent`` token alone is not enough, but a page about LLM agents
    using tools, agent memory, MCP, LangGraph or a named benchmark passes.
    """
    normalised = _normalise(text)
    if not normalised:
        return False
    if _has_strong_anchor(normalised):
        return True
    if _has_agent_cooccurrence(normalised):
        return True
    return _is_exact_ambiguous(normalised)


def scope_query(query: str) -> str:
    """Add Agentic AI context to a retrieval query that lacks it."""
    cleaned = " ".join((query or "").split())
    if not cleaned:
        return DOMAIN_QUERY_SUFFIX
    if is_domain_relevant(cleaned):
        return cleaned
    return f"{cleaned} {DOMAIN_QUERY_SUFFIX}"


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------
def _track_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


def canonical_track(name: str) -> DomainTrack | None:
    """Map a loose track name onto the fixed Agentic AI taxonomy."""
    key = _track_key(name)
    if not key:
        return None
    if key in _BY_KEY:
        return _BY_KEY[key]
    if key in _TRACK_ALIASES:
        return _BY_KEY[_TRACK_ALIASES[key]]

    # A longer name such as ``agent_memory`` or ``multi_agent_orchestration``
    # should still land on the right canonical track.
    for alias, target in _TRACK_ALIASES.items():
        if _has_phrase(key.replace("_", " "), alias.replace("_", " ")):
            return _BY_KEY[target]
    for track in DOMAIN_TRACKS:
        if _has_phrase(key.replace("_", " "), track.key):
            return track
    return None


def track_title(track: str, topic: str = "") -> str:
    """Human-readable title for a canonical or custom track."""
    canonical = canonical_track(track)
    if canonical is not None:
        return canonical.title
    return (track or "").replace("_", " ").replace("-", " ").title()


def track_menu() -> str:
    """Render the canonical taxonomy for the Director prompt."""
    return "\n".join(
        f"- {track.key}: {track.title} - {track.objective}" for track in DOMAIN_TRACKS
    )


def domain_protocol() -> str:
    """The domain-lock block included in every agent prompt."""
    return f"""\
DOMAIN LOCK - AGENTIC AI ONLY (mandatory):
- Mars researches exactly one domain: **{DOMAIN}** - {DOMAIN_DESCRIPTION}
- Every question, search query, source and finding must be about agentic AI.
  Do not drift into general AI/ML, other sciences, politics, history or
  unrelated industries.
- If a topic, source or instruction is outside Agentic AI, do not pursue it.
  Record "out of domain" as an evidence gap instead.
- Adjacent background is allowed only when it directly explains an agentic-AI
  capability, benchmark, risk or deployment pattern.
- Use agentic-AI vocabulary: agents, tool use, function calling, planning,
  memory, orchestration, evaluation, guardrails and deployment.\
"""
