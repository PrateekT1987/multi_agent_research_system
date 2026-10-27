"""Deterministic, offline LLM used only by the test-suite.

This provider is intentionally *not* part of the production package.  It speaks
the same CrewAI ``BaseLLM`` protocol as the live provider so the pipeline, the
native tool-calling loop and the report renderer can be exercised hermetically.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import Field, PrivateAttr, model_validator

from mars.llm import FunctionCall, ToolCall, _LLMBase

__all__ = ["MockLLM"]


# ---------------------------------------------------------------------------
# Offline / deterministic provider
# ---------------------------------------------------------------------------
_TRACK_HINTS = {
    "foundations": r"\b(foundations?|origins?|history|historical|evolution|timeline|precursors?|past)\b",
    "architectures": (
        r"\b(architectures?|orchestration|agent loops?|planning|memory|"
        r"tool use|tool calling|function calling|multi[- ]agent)\b"
    ),
    "capabilities": (
        r"\b(capabilit\w*|benchmarks?|evaluat\w*|evals?|reliability|"
        r"state of the art|current|performance)\b"
    ),
    "applications": r"\b(applications?|use cases?|deploy\w*|adoption|enterprise|case studies)\b",
    "safety": r"\b(safety|security|governance|policy|regulation|risks?|guardrails?|oversight)\b",
    "ecosystem": r"\b(ecosystem|frameworks?|protocols?|vendors?|economics|market)\b",
    "future": r"\b(future|forecast(s|ing|ed)?|outlook|anticipat\w*|roadmaps?|next|trajector\w*|horizon)\b",
}
_TRACK_WORDS = tuple(_TRACK_HINTS)
# The task templates name the track in bold ("the **capabilities** track"),
# which is far more reliable than counting keywords across a whole prompt.
_TRACK_BOLD_RE = re.compile(
    r"\*\*(" + "|".join(_TRACK_WORDS) + r")\*\*", re.IGNORECASE
)
_TRACK_LABEL_RE = re.compile(
    r"\btrack:\s*(" + "|".join(_TRACK_WORDS) + r")\b", re.IGNORECASE
)

# The questions and search queries the offline director hands to each track.
_MOCK_QUESTIONS: dict[str, list[str]] = {
    "foundations": [
        "What are the intellectual and technical ancestors of today's agents?",
        "Which publications and systems marked each transition?",
        "How did earlier failures shape current agent design?",
    ],
    "architectures": [
        "How is the modern agent loop structured?",
        "Which memory, planning and tool-use patterns are standard?",
        "How do multi-agent systems coordinate and delegate?",
    ],
    "capabilities": [
        "Which agent capabilities are demonstrated today?",
        "Which benchmarks measure agent reliability?",
        "Where do agents still fail on realistic multi-step tasks?",
    ],
    "applications": [
        "Where are agentic systems deployed in production?",
        "Which workflows change most when an agent is introduced?",
        "What does the evidence say about deployment impact?",
    ],
    "safety": [
        "Which security risks are specific to tool-using agents?",
        "How does prompt injection affect agentic systems?",
        "What oversight and governance mechanisms exist?",
    ],
    "ecosystem": [
        "Which frameworks and protocols dominate agent development?",
        "How are agent vendors and platforms positioned?",
        "What does agent deployment cost, and who pays?",
    ],
    "future": [
        "Which trajectories have the strongest supporting evidence?",
        "What are the binding technical and economic constraints?",
        "What would have to be true for the optimistic scenario?",
    ],
}

_MOCK_QUERIES: dict[str, list[str]] = {
    "foundations": ["agentic AI history milestones", "ReAct agent paper origins"],
    "architectures": ["LLM agent architecture tool use", "multi-agent orchestration patterns"],
    "capabilities": ["AI agent benchmarks reliability", "agent evaluation SWE-bench WebArena"],
    "applications": ["agentic AI production deployments", "coding agents enterprise adoption"],
    "safety": ["agent prompt injection security", "agentic AI governance guardrails"],
    "ecosystem": ["agent frameworks MCP A2A", "agentic AI deployment economics"],
    "future": ["AI agent forecasts roadmap", "agentic AI open problems trajectory"],
}


class MockLLM(_LLMBase):
    """A deterministic provider used for tests, demos and offline runs.

    The mock is role aware: it inspects the system prompt to work out which
    agent is speaking and returns a plausible, well-formed response for that
    role.  When it has access to the research context it cites the sources
    that are actually registered, so the grounding audit behaves exactly as it
    would in a live run.
    """

    llm_type: str = "mock"
    model: str = "mock/deterministic"
    temperature: float = 0.0
    context: Any = Field(default=None, exclude=True)

    _mock_tool_calls: int = PrivateAttr(default=0)

    @model_validator(mode="before")
    @classmethod
    def _inject_default_model(cls, data: Any) -> Any:
        """CrewAI's base validator insists on an explicit model name.

        The mock provider has a sensible default, so fill it in before the
        parent validator runs (subclass validators run first).
        """
        if isinstance(data, dict) and not data.get("model"):
            default = cls.model_fields["model"].default
            if default:
                data = {**data, "model": default}
        return data

    def supports_function_calling(self) -> bool:
        """The mock drives CrewAI's native tool loop, like the live provider.

        Returning ``True`` means CrewAI hands the tool schemas to
        :meth:`call` and feeds the results back, instead of asking the model
        to emit ``Action:`` lines.  The offline path therefore exercises the
        same code path as a live run.
        """
        return True

    def call(
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: type[Any] | None = None,
    ) -> str | Any:
        rendered = self._render(messages)
        system_text = "\n".join(
            str(m.get("content", "")) for m in rendered if m.get("role") == "system"
        )
        user_text = "\n".join(
            str(m.get("content", "")) for m in rendered if m.get("role") == "user"
        )
        self.usage.record(model=self.model, prompt_tokens=0, completion_tokens=0, role=self.label)

        if response_model is None and tools:
            tool_call = self._mock_tool_call(rendered, tools, system_text, user_text)
            if tool_call is not None:
                return [tool_call]
        tool_material = "\n".join(
            str(m.get("content", "")) for m in rendered if m.get("role") == "tool"
        )
        payload = self.reply(system_text, user_text, tool_material=tool_material)
        if response_model is not None:
            from mars.schemas import parse_json_block

            try:
                return response_model.model_validate(parse_json_block(payload))
            except Exception:
                return response_model.model_validate(json.loads(payload))
        return payload

    # --------------------------------------------------------------- helpers
    def _mock_tool_call(
        self,
        rendered: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_text: str,
        user_text: str,
    ) -> ToolCall | None:
        """Ask for one tool call, the way a real researcher would.

        The offline provider still drives CrewAI's native tool loop: the first
        call requests a retrieval tool, the tool result comes back in the next
        request, and the mock keeps going until it has used its research
        budget.  Only then does it answer with the material it gathered.

        The tools are tried in the order a careful researcher would use them:
        the curated corpus first (cheap and citable), then live search.
        """
        role = self._detect_role(system_text, user_text)
        if role not in {"researcher", "gap_researcher"}:
            return None

        names = [
            (tool.get("function") or {}).get("name")
            for tool in tools
            if isinstance(tool, dict)
        ]
        names = [name for name in names if name]
        if not names:
            return None

        used = {
            str(message.get("name") or "")
            for message in rendered
            if message.get("role") == "tool"
        }
        used.discard("")

        preferred = ("query_corpus", "arxiv_search", "web_search", "wikipedia_lookup")
        candidates = [name for name in preferred if name in names and name not in used]
        if not candidates:
            return None  # the research budget is spent - answer now

        target = candidates[0]

        topic = self.context.settings.topic if self.context is not None else "agentic AI"
        track = self._detect_track(f"{system_text}\n{user_text}")
        arguments: dict[str, Any] = {"query": f"{topic} {track}"}
        if target == "web_search":
            arguments = {"query": f"{topic} {track}", "max_results": 5}
        elif target == "arxiv_search":
            arguments = {"query": f"agentic AI {track}", "max_results": 5}
        elif target == "wikipedia_lookup":
            arguments = {"query": "AI agent"}

        self._mock_tool_calls += 1
        return ToolCall(
            id=f"call_mock_{self._mock_tool_calls}",
            function=FunctionCall(name=target, arguments=json.dumps(arguments)),
        )

    def reply(
        self,
        system_text: str,
        user_text: str,
        sources: list[tuple[str, str, str]] | None = None,
        *,
        material: str | None = None,
        tool_material: str | None = None,
    ) -> str:
        """Produce a deterministic reply for the given prompt.

        The reply is derived from the material that is actually in the prompt
        (the tool output for a researcher, the dossier for an analyst, the
        findings for a writer, the drafts for an editor) so that the offline
        run reproduces the same data flow as a live one.

        Exposed publicly so that test doubles (for example a fake OpenRouter
        server) can reuse exactly the same behaviour as the offline provider.
        """
        role = self._detect_role(system_text, user_text)
        track = self._detect_track(f"{system_text}\n{user_text}")
        prompt_material = user_text if material is None else material

        if sources is None:
            if role in {"researcher", "gap_researcher"}:
                sources = self._prompt_sources(tool_material or "")
            else:
                sources = self._prompt_sources(prompt_material)
            if not sources:
                sources = self._corpus_sources()

        if role == "director":
            return self._mock_plan(track)
        if role == "researcher":
            return self._mock_dossier(track, sources)
        if role == "analyst":
            return self._mock_analysis(track, sources)
        if role == "writer":
            return self._mock_section(track, sources, prompt_material)
        if role == "editor":
            return self._mock_report(sources, prompt_material)
        if role == "auditor":
            return self._mock_audit(prompt_material)
        return self._mock_dossier(track, sources)
    def _detect_role(self, system_text: str, user_text: str) -> str:
        """Work out which agent is speaking.

        The markers below come from Mars' own task templates and are unique
        to each stage, so detection does not depend on the tool descriptions
        or the agent backstory (which may legitimately mention other roles).
        """
        haystack = f"{system_text}\n{user_text}".lower()
        markers = (
            ("auditor", "audit the draft report below against the source registry"),
            ("editor", "assemble the final agentic ai report on"),
            ("writer", "section of an agentic ai research report on"),
            ("analyst", "convert it into structured findings"),
            ("researcher", "track of an agentic ai report on"),
            ("director", "you are the research director"),
        )
        for role, marker in markers:
            if marker in haystack:
                return role

        if "research director" in haystack or "research plan" in haystack:
            return "director"
        if "fact-check" in haystack or "auditor" in haystack:
            return "auditor"
        if "editor" in haystack:
            return "editor"
        if "writer" in haystack:
            return "writer"
        if "analyst" in haystack:
            return "analyst"
        return "researcher"

    def _detect_track(self, text: str) -> str:
        """Work out which track a prompt is about.

        Explicit markers always win over keyword counting: the plans and the
        finding bundles name their track outright, and the task templates name
        it in bold.  Counting keywords over a whole prompt is unreliable -
        a source title that mentions an earlier era is enough to mislead it.
        """
        lowered = text.lower()

        for track in _TRACK_WORDS:
            if f'"track": "{track}"' in lowered:
                return track

        bold = _TRACK_BOLD_RE.search(text)
        if bold:
            return bold.group(1).lower()

        labelled = _TRACK_LABEL_RE.search(text)
        if labelled:
            return labelled.group(1).lower()

        best = ("history", 0)
        for track, pattern in _TRACK_HINTS.items():
            score = len(re.findall(pattern, lowered))
            if score > best[1]:
                best = (track, score)
        return best[0]

    def _corpus_sources(self) -> list[tuple[str, str, str]]:
        """Return ``(source_id, title, snippet)`` for registered sources."""
        if self.context is None:
            return []
        store = self.context.evidence
        out: list[tuple[str, str, str]] = []
        for source in store.sources():
            out.append((source.id, source.title, source.snippet))
        return out

    def _prompt_sources(self, text: str) -> list[tuple[str, str, str]]:
        """Resolve the ``[S#]`` ids mentioned in ``text`` against the registry.

        This is the offline equivalent of a live model reading the ids out of
        the material it was given: only sources that were actually handed to
        the agent can be cited.
        """
        if self.context is None or not text:
            return []
        store = self.context.evidence
        out: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for source_id in store.extract_citations(text):
            if source_id in seen:
                continue
            seen.add(source_id)
            source = store.get(source_id)
            if source is None:
                continue
            out.append((source.id, source.title, source.snippet))
        return out

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        """Return the content between ``<tag>`` and ``</tag>``."""
        lowered = text.lower()
        start = lowered.find(f"<{tag}>")
        if start == -1:
            return ""
        start += len(tag) + 2
        end = lowered.find(f"</{tag}>", start)
        if end == -1:
            return text[start:]
        return text[start:end].strip()

    def _extract_findings(self, material: str) -> list[dict[str, Any]]:
        """Pull the findings out of the ``<findings>`` block of a prompt."""
        from mars.schemas import parse_json_block

        block = self._extract_tag(material, "findings")
        if not block:
            return []
        try:
            payload = parse_json_block(block)
        except ValueError:
            return []
        if isinstance(payload, dict):
            findings = payload.get("findings")
            if isinstance(findings, list):
                return [f for f in findings if isinstance(f, dict)]
        return []

    @staticmethod
    def _extract_sections(material: str) -> dict[str, str]:
        """Split the ``<sections>`` block into ``{heading: body}``."""
        lowered = material.lower()
        start = lowered.find("<sections>")
        if start == -1:
            return {}
        start += len("<sections>")
        end = lowered.find("</sections>", start)
        block = material[start : end if end != -1 else len(material)]

        sections: dict[str, str] = {}
        current: str | None = None
        buffer: list[str] = []
        for line in block.splitlines():
            if line.startswith("## "):
                if current is not None:
                    sections[current] = "\n".join(buffer).strip()
                current = line[3:].strip()
                buffer = []
            elif current is not None:
                buffer.append(line)
        if current is not None:
            sections[current] = "\n".join(buffer).strip()
        return sections

    @staticmethod
    def _extract_structure(material: str) -> list[str]:
        """Return the section headings the editor was told to produce.

        The structure block is anchored on its intro line so that the drafted
        sections (which also carry ``##`` headings) are not mistaken for it.
        """
        lines = material.splitlines()
        start = None
        for index, line in enumerate(lines):
            if line.strip().upper().startswith("STRUCTURE"):
                start = index + 1
                break
        if start is None:
            return []

        headings: list[str] = []
        for line in lines[start:]:
            stripped = line.strip()
            if not stripped:
                if headings:
                    break
                continue
            if stripped.startswith("## "):
                heading = stripped[3:].strip()
                if heading and heading not in headings:
                    headings.append(heading)
                continue
            if stripped.startswith("# "):
                continue
            break
        return headings

    def _respond(self, system_text: str, user_text: str, rendered: list[dict[str, Any]]) -> str:
        return self.reply(system_text, user_text)

    # ---------------------------------------------------------------- mocks
    def _mock_plan(self, track: str) -> str:
        topic = self.context.settings.topic if self.context is not None else "agentic AI"
        all_tracks = (
            list(self.context.settings.tracks)
            if self.context is not None
            else ["foundations", "architectures", "capabilities", "applications", "safety", "future"]
        )
        tracks_payload = [
            {
                "track": name,
                "objective": f"Establish the {name} of {topic}.",
                "key_questions": _MOCK_QUESTIONS.get(name, _MOCK_QUESTIONS["foundations"]),
                "search_queries": _MOCK_QUERIES.get(name, [f"{topic} {name}"]),
                "keywords": [name, topic, "agents"],
                "pitfalls": ["uncited claims", "anecdotal evidence presented as trend"],
            }
            for name in all_tracks
        ]
        plan = {
            "topic": topic,
            "domain": "Agentic AI",
            "framing": "A multi-track investigation of the Agentic AI subtopic.",
            "tracks": tracks_payload,
        }
        return json.dumps(plan, indent=2)

    def _mock_dossier(self, track: str, sources: list[tuple[str, str, str]]) -> str:
        """The researcher's dossier, in the shape the task asks for."""
        if not sources:
            return (
                "## Findings\n"
                "- No sources were retrieved. Note this coverage gap explicitly in the "
                "section and flag it for the auditor.\n"
                "\n"
                "## Evidence gaps\n"
                "- The retrieval tools returned nothing usable for this track."
            )
        picked = sources[:8]
        lines = ["## Findings", ""]
        for source_id, title, snippet in picked:
            detail = " ".join((snippet or "").split())[:220]
            lines.append(f"- {title}: {detail} [{source_id}]")
        lines.extend(["", "## Timeline", ""])
        for source_id, title, _snippet in picked[:3]:
            source = self._source(source_id)
            date = (source.published if source else None) or "undated"
            lines.append(f"- {date}: {title} [{source_id}]")
        lines.extend(
            [
                "",
                "## Disagreements & uncertainty",
                "",
                "- The retrieved sources do not agree on how far the field has "
                "matured; treat single-source statements as provisional.",
                "",
                "## Evidence gaps",
                "",
                "- Primary sources for the most recent period are thin in this "
                "retrieval; the auditor should check the claims marked provisional.",
                "",
                "## Sources used",
                "",
            ]
        )
        for source_id, title, _snippet in picked:
            lines.append(f"- [{source_id}] {title}")
        return "\n".join(lines)

    def _source(self, source_id: str) -> Any:
        if self.context is None:
            return None
        return self.context.evidence.get(source_id)

    def _mock_analysis(self, track: str, sources: list[tuple[str, str, str]]) -> str:
        """Turn the dossier into structured findings for one track."""
        findings: list[dict[str, Any]] = []
        for source_id, title, snippet in sources[:6]:
            detail = " ".join((snippet or "").split())[:200]
            findings.append(
                {
                    "text": f"{title}: {detail}",
                    "evidence": [source_id],
                    "confidence": 0.7,
                    "kind": "forecast" if track == "future" else "fact",
                }
            )
        if not findings:
            findings.append(
                {
                    "text": "No evidence was gathered for this track.",
                    "evidence": [],
                    "confidence": 0.1,
                    "kind": "fact",
                }
            )
        timeline = []
        for source_id, title, _snippet in sources[:3]:
            source = self._source(source_id)
            timeline.append(
                {
                    "date": (source.published if source else None) or "undated",
                    "event": title,
                    "significance": "Documents a stage in the development of the field.",
                    "source_ids": [source_id],
                }
            )
        return json.dumps(
            {
                "track": track,
                "summary": (
                    f"The {track} track rests on {len(findings)} sourced findings drawn "
                    f"from {len(sources)} registered sources."
                ),
                "findings": findings,
                "timeline": timeline,
                "open_questions": [
                    f"What does this imply for the {track} of the field beyond the "
                    "sources retrieved here?"
                ],
            },
            indent=2,
        )

    def _mock_section(
        self,
        track: str,
        sources: list[tuple[str, str, str]],
        material: str = "",
    ) -> str:
        """Write one section from the findings the analyst handed over."""
        titles = {
            "foundations": "Foundations and Origins",
            "architectures": "Agent Architectures and Orchestration",
            "capabilities": "Capabilities, Benchmarks and Evaluation",
            "applications": "Applications and Deployment",
            "safety": "Safety, Security and Governance",
            "ecosystem": "Ecosystem, Adoption and Economics",
            "future": "Future Trajectory",
        }
        heading = self._section_heading(material) or titles.get(track, track.title())
        findings = self._extract_findings(material)

        anchor = ""
        for finding in findings:
            evidence = finding.get("evidence") or []
            if evidence:
                anchor = f"[{evidence[0]}]"
                break

        lines = [f"## {heading}", ""]
        if not findings:
            lines.append(
                "This track could not be sourced from the available material; the gap "
                "is recorded rather than filled with speculation."
            )
            return "\n".join(lines)

        lines.append(
            f"The {track} track resolves into {len(findings)} load-bearing findings, "
            "each resting on at least one registered source. Read together they "
            "describe a field that is moving from hand-coded behaviour towards "
            f"model-driven systems that choose their own actions. {anchor}".rstrip()
        )
        lines.append("")
        lines.append("### What the evidence shows")
        lines.append("")
        for finding in findings:
            text = str(finding.get("text", "")).strip()
            if not text:
                continue
            ids = "".join(f"[{cid}]" for cid in (finding.get("evidence") or []))
            lines.append(f"- {text} {ids}".rstrip())
        lines.append("")
        lines.append("### What this means")
        lines.append("")
        lines.append(
            "Taken together, these findings describe a field that is converging on "
            "standardised agent architectures while its reliability and governance "
            f"questions remain open. {anchor}".rstrip()
        )
        return "\n".join(lines)

    @staticmethod
    def _section_heading(material: str) -> str:
        """Read the requested section title out of the writer prompt."""
        for line in material.splitlines():
            stripped = line.strip()
            if stripped.startswith("SECTION TITLE:"):
                return stripped[len("SECTION TITLE:") :].strip()
        return ""

    def _mock_report(
        self,
        sources: list[tuple[str, str, str]],
        material: str = "",
    ) -> str:
        """Assemble the final report from the section drafts."""
        title = self._report_title(material)
        structure = self._extract_structure(material)
        drafts = self._extract_sections(material)
        anchor = f"[{sources[0][0]}]" if sources else ""

        lines = [f"# {title}", "", "## Executive Summary", ""]
        if sources:
            lines.append(
                f"This report synthesises {len(sources)} registered sources into a "
                "narrative covering origins, current practice, and forward trajectory. "
                "The strongest through-line is the shift from symbolic control to "
                f"model-driven tool use. [{sources[0][0]}]"
            )
        else:
            lines.append("No sources were available; this report is a structural stub.")
        lines.append("")
        track_headings = [
            heading
            for heading in structure
            if "timeline" not in heading.lower()
            and "open question" not in heading.lower()
            and "implication" not in heading.lower()
            and "executive summary" not in heading.lower()
        ]

        for heading in track_headings:
            lines.append(f"## {heading}")
            lines.append("")
            body = self._match_draft(heading, drafts)
            if body:
                lines.append(body)
            else:
                for source_id, source_title, snippet in sources[:3]:
                    detail = " ".join((snippet or "").split())[:150]
                    lines.append(f"- {source_title}: {detail} [{source_id}]")
            lines.append("")

        for heading in structure:
            lowered = heading.lower()
            if "timeline" in lowered:
                lines.append(f"## {heading}")
                lines.append("")
                lines.append("| Date | Event | Why it mattered | Sources |")
                lines.append("| --- | --- | --- | --- |")
                for source_id, source_title, _snippet in sources[:5]:
                    source = self._source(source_id)
                    date = (source.published if source else None) or "undated"
                    lines.append(
                        f"| {date} | {source_title} | Marks a documented stage of "
                        f"the field | [{source_id}] |"
                    )
                lines.append("")
            elif "open question" in lowered:
                lines.append(f"## {heading}")
                lines.append("")
                lines.append("**What we still do not know**")
                lines.append("")
                lines.append(
                    "- The retrieval behind this run did not cover primary reporting "
                    "for the most recent period, so claims about the current state "
                    f"rest on secondary sources. {anchor}"
                )
                lines.append("")
                lines.append("**What we disagree about**")
                lines.append("")
                lines.append(
                    "- The retrieved sources disagree on how mature the field is; the "
                    "report keeps both readings rather than picking one. "
                    f"{anchor}"
                )
                lines.append("")
            elif "implication" in lowered:
                lines.append(f"## {heading}")
                lines.append("")
                for source_id, source_title, _snippet in sources[:4]:
                    lines.append(
                        "- Decision-makers tracking this field should follow "
                        f"{source_title} closely. [{source_id}]"
                    )
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _report_title(material: str) -> str:
        for line in material.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("##"):
                return stripped[2:].strip()
        return "Research Report"

    @staticmethod
    def _match_draft(heading: str, drafts: dict[str, str]) -> str:
        """Find the drafted section that belongs to ``heading``.

        The editor numbers its sections ("1. Origins and Evolution") while the
        writer drafts them unnumbered, so the numbering is stripped before
        matching.
        """
        wanted = heading.split(". ", 1)[-1].strip().lower()
        for draft_heading, body in drafts.items():
            candidate = draft_heading.split(". ", 1)[-1].strip().lower()
            if candidate == wanted:
                return body
        for draft_heading, body in drafts.items():
            if wanted and wanted in draft_heading.lower():
                return body
        return ""

    def _mock_audit(self, material: str = "") -> str:
        """Audit the draft the same way the live auditor is asked to.

        The grounding numbers are computed from the draft itself rather than
        being hard-coded, so the offline report's audit appendix agrees with
        the appendix the pipeline appends.
        """
        store = self.context.evidence if self.context is not None else None
        draft = self._extract_tag(material, "draft")

        if store is None or not draft:
            return json.dumps(
                {
                    "grounded_ratio": 0.9,
                    "cited_claims": 9,
                    "total_claims": 10,
                    "unsupported_claims": [],
                    "dangling_citations": [],
                    "uncited_paragraphs": [],
                    "gaps": [],
                    "verdict": "pass",
                    "notes": "Deterministic audit from the offline provider.",
                },
                indent=2,
            )

        ratio, cited, total = store.grounding_score(draft)
        dangling = store.dangling_citations(draft)
        uncited = store.uncited_paragraphs(draft)
        threshold = self.context.settings.grounding_threshold
        verdict = "pass" if ratio >= threshold and not dangling else "revise"

        gaps: list[str] = []
        if verdict == "revise":
            gaps.append(
                "Which of the uncited paragraphs above rest on a source that was "
                "retrieved but not cited?"
            )

        return json.dumps(
            {
                "grounded_ratio": round(ratio, 3),
                "cited_claims": cited,
                "total_claims": total,
                "unsupported_claims": [],
                "dangling_citations": dangling,
                "uncited_paragraphs": uncited,
                "gaps": gaps,
                "verdict": verdict,
                "notes": (
                    "Deterministic audit from the offline provider: the grounding "
                    "score is computed from the draft against the source registry."
                ),
            },
            indent=2,
        )

