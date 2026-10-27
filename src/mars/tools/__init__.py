"""Tool factory: builds the toolset handed to each agent."""

from __future__ import annotations

from crewai.tools import BaseTool

from ..config import Settings
from ..context import ResearchContext
from .notes import ListSourcesTool, ReadFindingsTool, RecordFindingTool
from .papers import ArxivSearchTool, HackerNewsTool, SemanticScholarTool, WikipediaTool
from .web import WebFetchTool, WebSearchTool

__all__ = [
    "build_toolset",
    "RESEARCH_TOOL_NAMES",
    "NETWORK_TOOL_NAMES",
]

RESEARCH_TOOL_NAMES = (
    "web_search",
    "fetch_page",
    "arxiv_search",
    "semantic_scholar_search",
    "wikipedia_lookup",
    "hackernews_search",
    "record_finding",
    "list_sources",
    "read_findings",
)

# Tools that reach out to the network.  They are withheld when network access is
# disabled so a run can be guaranteed to make no outbound requests.
NETWORK_TOOL_NAMES = frozenset(
    {
        "web_search",
        "fetch_page",
        "arxiv_search",
        "semantic_scholar_search",
        "wikipedia_lookup",
        "hackernews_search",
    }
)


def build_toolset(
    context: ResearchContext,
    *,
    track: str = "general",
    include: tuple[str, ...] | None = None,
    settings: Settings | None = None,
) -> list[BaseTool]:
    """Create the tools available to a research agent.

    Parameters
    ----------
    context:
        Shared runtime context (settings, HTTP client, evidence store).
    track:
        The research track the agent is working on. Recorded on findings so the
        ledger stays organised.
    include:
        Optional whitelist of tool names. Defaults to the full research set.
    settings:
        Overrides the context's settings.
    """
    settings = settings or context.settings
    wanted = set(include or RESEARCH_TOOL_NAMES)
    if not settings.allow_network:
        # Offline runs must not even be offered tools that reach the network.
        wanted -= NETWORK_TOOL_NAMES

    tools: list[BaseTool] = []
    if "web_search" in wanted:
        tools.append(WebSearchTool(context=context))
    if "fetch_page" in wanted:
        tools.append(WebFetchTool(context=context))
    if "arxiv_search" in wanted:
        tools.append(ArxivSearchTool(context=context))
    if "semantic_scholar_search" in wanted:
        tools.append(SemanticScholarTool(context=context))
    if "wikipedia_lookup" in wanted:
        tools.append(WikipediaTool(context=context))
    if "hackernews_search" in wanted:
        tools.append(HackerNewsTool(context=context))
    if "record_finding" in wanted:
        tools.append(RecordFindingTool(context=context, track=track))
    if "list_sources" in wanted:
        tools.append(ListSourcesTool(context=context))
    if "read_findings" in wanted:
        tools.append(ReadFindingsTool(context=context))
    return tools
