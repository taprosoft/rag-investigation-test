"""Inline citation markers, following OpenAI's citation-formatting convention.

https://developers.openai.com/api/docs/guides/citation-formatting

The model writes ordinary prose and drops a marker straight after the sentence a source
supports::

    Funds left the account within minutes.\\ue200cite\\ue202E1\\ue202L2\\ue201

Private-use delimiters are used precisely because they cannot collide with anything a
model would legitimately write, so extraction is exact rather than heuristic. Two things
follow from that, and both matter here:

* citations land *where the claim is*, not in a footnote list bolted on afterwards, so a
  reader can see which half of a sentence is sourced;
* the locator addresses numbered lines of a passage, so "show me the source" means
  highlighting one sentence rather than a whole paragraph.

Every citation is then checked against the evidence actually supplied. A model that
invents a label, or points past the end of a passage, produces an *unverified* citation
that the UI flags — the failure is surfaced, not silently rendered as grounded.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from detective.core.models import Citation, Evidence

CITATION_START = ""
CITATION_DELIMITER = ""
CITATION_STOP = ""

SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
LINE_LOCATOR_RE = re.compile(r"^L\d+(?:-L\d+)?$")

_TOKEN_RE = re.compile(
    rf"{re.escape(CITATION_START)}"
    rf"(?P<family>cite)"
    rf"{re.escape(CITATION_DELIMITER)}"
    rf"(?P<body>.*?)"
    rf"{re.escape(CITATION_STOP)}",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class RawCitation:
    """A marker as it appeared in the model's text, before validation."""

    source_ids: tuple[str, ...]
    locator: str | None
    start: int
    end: int


def extract_citations(text: str) -> list[RawCitation]:
    """Pull every well-formed citation marker out of ``text``, keeping its offsets.

    Malformed markers are dropped rather than raised on: a half-written marker is a
    generation artefact, and losing one citation is better than losing the report.
    """
    citations: list[RawCitation] = []
    for match in _TOKEN_RE.finditer(text):
        parts = [part.strip() for part in match.group("body").split(CITATION_DELIMITER)]
        parts = [part for part in parts if part]
        if not parts:
            continue
        locator = parts.pop() if LINE_LOCATOR_RE.fullmatch(parts[-1]) else None
        if not parts or any(not SOURCE_ID_RE.fullmatch(part) for part in parts):
            continue
        citations.append(
            RawCitation(
                source_ids=tuple(parts), locator=locator, start=match.start(), end=match.end()
            )
        )
    return citations


def strip_citations(text: str, citations: Sequence[RawCitation]) -> str:
    """Remove markers from ``text``, right to left so earlier offsets stay valid."""
    clean = text
    for citation in sorted(citations, key=lambda c: c.start, reverse=True):
        clean = clean[: citation.start] + clean[citation.end :]
    return clean


def parse_locator(locator: str | None, line_count: int) -> tuple[int, ...]:
    """Resolve ``L2`` or ``L2-L4`` to line numbers that exist in a passage.

    An empty result means the locator did not resolve, which is what makes the citation
    unverified. A citation with no locator at all is passage-level and cites every line.
    """
    if locator is None:
        return tuple(range(1, line_count + 1))
    if not LINE_LOCATOR_RE.fullmatch(locator):
        return ()
    bounds = [int(part[1:]) for part in locator.split("-")]
    first, last = (bounds[0], bounds[-1])
    if first > last:
        first, last = last, first
    resolved = tuple(n for n in range(first, last + 1) if 1 <= n <= line_count)
    return resolved


def parse_claim(text: str, evidence: Sequence[Evidence]) -> tuple[str, tuple[Citation, ...]]:
    """Split model prose into clean text plus validated citations with their offsets."""
    by_label = {item.label: item for item in evidence}
    raw = extract_citations(text)
    clean = strip_citations(text, raw)

    citations: list[Citation] = []
    removed = 0
    for marker in sorted(raw, key=lambda c: c.start):
        offset = marker.start - removed
        removed += marker.end - marker.start
        for source_id in marker.source_ids:
            source = by_label.get(source_id)
            lines = parse_locator(marker.locator, len(source.lines)) if source else ()
            citations.append(
                Citation(
                    label=source_id,
                    locator=marker.locator,
                    lines=lines,
                    verified=source is not None and bool(lines),
                    offset=offset,
                )
            )
    return clean.strip(), tuple(citations)


def format_evidence(evidence: Sequence[Evidence], *, numbered: bool = True) -> str:
    """Render passages for the synthesiser with stable ids and numbered lines."""
    blocks: list[str] = []
    for item in evidence:
        body = (
            "\n".join(f"L{n}: {line}" for n, line in enumerate(item.lines, start=1))
            if numbered
            else item.chunk.text
        )
        blocks.append(f"[{item.label}] source: {item.doc_id}\n{body}")
    return "\n\n".join(blocks)


def marker(label: str, locator: str | None = None) -> str:
    """Build a citation marker. Used by tests and by the offline stand-in synthesiser."""
    body = label if locator is None else f"{label}{CITATION_DELIMITER}{locator}"
    return f"{CITATION_START}cite{CITATION_DELIMITER}{body}{CITATION_STOP}"
