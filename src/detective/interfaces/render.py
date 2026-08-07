"""Rendering an investigation as Markdown (CLI, archive) and HTML (web UI).

The HTML side is where the citation UX lives. Each citation becomes a chip sitting exactly
where the model placed its marker, carrying the DOM id of the sentence it cites; each
evidence passage is rendered with its sentences individually addressable. Clicking a chip
therefore highlights one sentence, not a paragraph — and because the mapping is computed
here from resolved line numbers rather than searched for in the browser, the behaviour is
deterministic and unit-testable.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence

from detective.core.models import Citation, Evidence, Finding, Investigation, Report


def _line_id(label: str, number: int) -> str:
    return f"line-{label}-{number}"


def _citations(report: Report) -> Iterable[Citation]:
    for claim in report.claims:
        yield from claim.citations


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------


def report_markdown(investigation: Investigation) -> str:
    """Plain-text report for the CLI and for archival."""
    report = investigation.report
    lines = ["# Investigation report", "", f"**Question:** {investigation.question}", ""]
    if report is None:
        return "\n".join([*lines, "_No report was generated._"])

    lines += [
        f"_Retrieval mode: {investigation.mode}._",
        "",
        "## Summary",
        "",
        report.summary or "_none_",
        "",
    ]

    if report.timeline:
        lines += ["## Timeline", ""]
        lines += [f"{i}. {_claim_markdown(c)}" for i, c in enumerate(report.timeline, 1)]
        lines.append("")

    if report.findings:
        lines += ["## Findings", ""]
        lines += [f"- {_claim_markdown(c)}" for c in report.findings]
        lines.append("")

    lines += ["## Evidence", ""]
    for item in investigation.evidence:
        lines += [
            f"**[{item.label}] {item.doc_id}** — relevance {item.rerank_score:.2f} "
            f"(round {item.round_index}, query: {item.query!r})",
            "",
        ]
        lines += [f"> L{n}: {line}" for n, line in enumerate(item.lines, start=1)]
        lines += ["", f"_Why included:_ {item.rationale}", ""]

    if report.excluded:
        lines += ["## Retrieved but excluded", ""]
        lines += [f"- **{label or 'unlabelled'}** — {reason}" for label, reason in report.excluded]
        lines.append("")

    if report.open_questions:
        lines += ["## Open questions", ""]
        lines += [f"- {q}" for q in report.open_questions]
        lines.append("")

    lines += [
        "---",
        f"_Citations: {report.grounding_rate:.0%} of claims resolve to a specific "
        f"line of a supplied passage._",
    ]
    return "\n".join(lines)


def _claim_markdown(finding: Finding) -> str:
    """Re-insert citations as bracketed references at the offsets the model chose."""
    text = finding.statement
    for citation in sorted(finding.citations, key=lambda c: c.offset, reverse=True):
        cut = min(citation.offset, len(text))
        chip = f" [{citation.display}]" + ("" if citation.verified else "⚠")
        text = text[:cut] + chip + text[cut:]
    return " ".join(text.split())


def retrieval_trace_markdown(investigation: Investigation) -> str:
    """How the answer was reached — one section per search round."""
    lines: list[str] = []
    for round_result in investigation.rounds:
        lines += [
            f"### Round {round_result.index}",
            f"- **Query:** `{round_result.query}`",
            f"- **Why:** {round_result.reason}",
        ]
        if round_result.admitted:
            admitted = ", ".join(
                f"{e.label} ({e.chunk.chunk_id}, {e.rerank_score:.2f})"
                for e in round_result.admitted
            )
            lines.append(f"- **Admitted:** {admitted}")
        if round_result.rejected:
            lines.append("- **Rejected:**")
            lines += [
                f"    - `{chunk_id}` ({score:.2f}) — {reason}"
                for chunk_id, score, reason in round_result.rejected
            ]
        lines.append(
            f"- **Verdict:** {'sufficient' if round_result.sufficient else 'continue'}"
            + (f" — still missing: {round_result.gap}" if round_result.gap else "")
        )
        lines.append("")
    return "\n".join(lines) or "_No retrieval rounds recorded._"


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------


def render_panels(investigation: Investigation) -> tuple[str, str]:
    """Build the ``(report_html, evidence_html)`` pair shown side by side in the UI."""
    report = investigation.report
    if report is None:
        return "<p class='muted'>No report generated.</p>", ""
    return _report_html(investigation, report), _evidence_html(investigation.evidence, report)


def _report_html(investigation: Investigation, report: Report) -> str:
    parts = [
        "<div class='report'>",
        "<h3>Summary</h3><p>",
        html.escape(report.summary) or "<em>none</em>",
        "</p>",
    ]
    if report.timeline:
        parts.append("<h3>Timeline</h3><ol>")
        parts += [f"<li>{_claim_html(c)}</li>" for c in report.timeline]
        parts.append("</ol>")
    if report.findings:
        parts.append("<h3>Findings</h3><ul>")
        parts += [f"<li>{_claim_html(c)}</li>" for c in report.findings]
        parts.append("</ul>")
    if report.excluded:
        parts.append("<h3>Retrieved but excluded</h3><ul>")
        parts += [
            f"<li><strong>{html.escape(label or 'unlabelled')}</strong> — "
            f"{html.escape(reason)}</li>"
            for label, reason in report.excluded
        ]
        parts.append("</ul>")
    if report.open_questions:
        parts.append("<h3>Open questions</h3><ul>")
        parts += [f"<li>{html.escape(q)}</li>" for q in report.open_questions]
        parts.append("</ul>")
    parts.append(
        f"<p class='muted'>{len(investigation.evidence)} passages cited · "
        f"{report.grounding_rate:.0%} of claims resolve to a specific source line · "
        f"{len(investigation.rounds)} search round(s) · {html.escape(investigation.mode)} mode"
        "</p></div>"
    )
    return "".join(parts)


def _claim_html(finding: Finding) -> str:
    """Escape the statement, then splice citation chips in at the model's offsets."""
    text = finding.statement
    pieces: list[str] = []
    cursor = 0
    for citation in sorted(finding.citations, key=lambda c: c.offset):
        cut = min(max(citation.offset, 0), len(text))
        pieces.append(html.escape(text[cursor:cut]))
        pieces.append(_citation_html(citation))
        cursor = cut
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def _citation_html(citation: Citation) -> str:
    target = _line_id(citation.label, citation.lines[0]) if citation.lines else ""
    classes = "cite" if citation.verified else "cite unverified"
    title = (
        f"Source: {citation.label}, line{'s' if len(citation.lines) > 1 else ''} "
        + ", ".join(f"L{n}" for n in citation.lines)
        if citation.verified
        else "This citation does not resolve to a supplied passage"
    )
    lines = ",".join(str(n) for n in citation.lines)
    return (
        f"<a class='{classes}' href='#{html.escape(citation.anchor)}' "
        f"data-anchor='{html.escape(citation.anchor)}' "
        f"data-lines='{html.escape(lines)}' data-label='{html.escape(citation.label)}' "
        f"data-target='{html.escape(target)}' title='{html.escape(title)}'>"
        f"{html.escape(citation.display)}{'' if citation.verified else ' ⚠'}</a>"
    )


def _evidence_html(evidence: Sequence[Evidence], report: Report) -> str:
    if not evidence:
        return "<p class='muted'>No evidence met the relevance bar.</p>"

    cited: dict[str, set[int]] = {item.label: set() for item in evidence}
    for citation in _citations(report):
        if citation.verified and citation.label in cited:
            cited[citation.label].update(citation.lines)

    cards: list[str] = []
    for item in evidence:
        sentences = "".join(
            f"<span class='line{' cited' if n in cited[item.label] else ''}' "
            f"id='{_line_id(item.label, n)}'>{html.escape(line)}</span> "
            for n, line in enumerate(item.lines, start=1)
        )
        cards.append(
            f"<div class='evidence' id='{html.escape(item.anchor_id)}'>"
            f"<div class='evidence-head'><span class='label'>{html.escape(item.label)}</span>"
            f"<span class='source'>{html.escape(item.doc_id)}</span>"
            f"<span class='score' title='cross-encoder relevance'>"
            f"{item.rerank_score:.2f}</span></div>"
            f"<p class='passage'>{sentences}</p>"
            f"<p class='muted'>Round {item.round_index} · query "
            f"<code>{html.escape(item.query)}</code><br>{html.escape(item.rationale)}</p>"
            "</div>"
        )
    return "".join(cards)
