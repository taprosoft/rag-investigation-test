"""Domain types.

The pipeline is deliberately transparent: every intermediate an operator might want to
audit — which query was tried, what was admitted, what was rejected and why — is a value
in one of these structures rather than a log line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Retriever = Literal["dense", "lexical", "fused", "rerank"]

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class Document:
    """One source file in the corpus."""

    doc_id: str
    path: Path
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A passage of a document; the unit of retrieval."""

    chunk_id: str
    doc_id: str
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with the score that put it in a result list."""

    chunk: Chunk
    score: float
    scorer: Retriever


@dataclass(frozen=True, slots=True)
class Evidence:
    """A passage the agent admitted into the case file.

    ``label`` is the stable id the model cites, and ``lines`` is the passage split into
    numbered sentences — the unit a citation locator addresses, and therefore the unit the
    UI can highlight.
    """

    label: str
    chunk: Chunk
    rerank_score: float
    round_index: int
    query: str
    rationale: str

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def anchor_id(self) -> str:
        """DOM id of this passage's card in the web UI; citations link to it."""
        return f"evidence-{self.label}"

    @property
    def lines(self) -> tuple[str, ...]:
        """Sentences of the passage, addressed as ``L1``, ``L2``, … by citations."""
        return tuple(part for part in _SENTENCE_BREAK.split(self.chunk.text.strip()) if part)

    def line_text(self, numbers: tuple[int, ...]) -> str:
        lines = self.lines
        return " ".join(lines[n - 1] for n in numbers if 1 <= n <= len(lines))


@dataclass(frozen=True, slots=True)
class Citation:
    """A claim's pointer back to specific lines of an evidence passage.

    ``verified`` records whether the label and locator actually resolve. An unresolved
    citation means the model invented a source or pointed past the end of a passage, and
    the UI marks the claim accordingly rather than silently presenting it as grounded.
    """

    label: str
    locator: str | None
    lines: tuple[int, ...]
    verified: bool
    offset: int = 0
    """Character position in the cleaned statement where the marker appeared."""

    @property
    def anchor(self) -> str:
        return f"evidence-{self.label}"

    @property
    def display(self) -> str:
        """Short label for the chip, normalised from the lines that actually resolved.

        Models write locators like ``L1-L1``; echoing that back verbatim is noise, so the
        resolved range is the source of truth for what the reader sees.
        """
        if not self.lines:
            return self.label
        span = (
            f"L{self.lines[0]}" if len(self.lines) == 1 else f"L{self.lines[0]}-L{self.lines[-1]}"
        )
        return f"{self.label}·{span}"


@dataclass(frozen=True, slots=True)
class Finding:
    """One claim in the report: prose with the citation markers already extracted."""

    statement: str
    citations: tuple[Citation, ...]

    @property
    def fully_grounded(self) -> bool:
        return bool(self.citations) and all(c.verified for c in self.citations)


@dataclass(frozen=True, slots=True)
class Report:
    """The generated investigation report."""

    summary: str
    findings: tuple[Finding, ...]
    timeline: tuple[Finding, ...]
    open_questions: tuple[str, ...]
    excluded: tuple[tuple[str, str], ...] = ()
    """``(label, reason)`` for passages retrieved but judged non-probative."""

    @property
    def claims(self) -> tuple[Finding, ...]:
        return self.timeline + self.findings

    @property
    def grounding_rate(self) -> float:
        if not self.claims:
            return 0.0
        return sum(c.fully_grounded for c in self.claims) / len(self.claims)


@dataclass(frozen=True, slots=True)
class RoundResult:
    """One iteration of the agentic retrieval loop."""

    index: int
    query: str
    reason: str
    candidates: tuple[ScoredChunk, ...]
    admitted: tuple[Evidence, ...]
    rejected: tuple[tuple[str, float, str], ...]
    """``(chunk_id, rerank_score, reason)`` for candidates the loop turned down."""
    sufficient: bool
    gap: str


@dataclass(slots=True)
class Investigation:
    """Everything produced for one detective question."""

    question: str
    mode: str = "agentic"
    rounds: list[RoundResult] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    report: Report | None = None

    @property
    def evidence_by_label(self) -> dict[str, Evidence]:
        return {e.label: e for e in self.evidence}

    @property
    def sources(self) -> list[str]:
        seen: dict[str, None] = {}
        for item in self.evidence:
            seen.setdefault(item.doc_id, None)
        return list(seen)
