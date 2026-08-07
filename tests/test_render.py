from __future__ import annotations

from detective.core.models import Chunk, Citation, Evidence, Finding, Investigation, Report
from detective.interfaces.render import (
    render_panels,
    report_markdown,
    retrieval_trace_markdown,
)

PASSAGE = (
    "Camera 4 stopped recording at 02:03. "
    "Engineers found the port had been disabled at the switch. "
    "No other camera lost coverage."
)


def evidence(label: str = "E1", text: str = PASSAGE) -> Evidence:
    return Evidence(
        label=label,
        chunk=Chunk(chunk_id="doc#0", doc_id="doc", index=0, text=text),
        rerank_score=0.87,
        round_index=1,
        query="camera outage",
        rationale="places the coverage gap",
    )


def citation(label: str = "E1", locator: str = "L1", *, verified: bool = True) -> Citation:
    return Citation(
        label=label,
        locator=locator,
        lines=(int(locator[1:]),) if verified else (),
        verified=verified,
        offset=len("Coverage was lost."),
    )


def report_with(*citations: Citation, statement: str = "Coverage was lost.") -> Report:
    return Report(
        summary="A camera lost coverage.",
        findings=(Finding(statement=statement, citations=citations),),
        timeline=(),
        open_questions=(),
    )


def investigation_with(report: Report, items: list[Evidence]) -> Investigation:
    return Investigation(question="What failed?", evidence=items, report=report)


class TestRenderPanels:
    def test_citation_chip_targets_the_cited_line(self) -> None:
        item = evidence()

        report_html, evidence_html = render_panels(
            investigation_with(report_with(citation("E1", "L2")), [item])
        )

        assert f"data-anchor='{item.anchor_id}'" in report_html
        assert "data-lines='2'" in report_html
        assert "data-target='line-E1-2'" in report_html
        assert f"id='{item.anchor_id}'" in evidence_html
        assert "id='line-E1-2'" in evidence_html
        assert "class='line cited' id='line-E1-2'" in evidence_html

    def test_uncited_lines_are_rendered_but_not_marked(self) -> None:
        _, evidence_html = render_panels(
            investigation_with(report_with(citation("E1", "L2")), [evidence()])
        )

        assert "class='line' id='line-E1-1'" in evidence_html
        assert "class='line' id='line-E1-3'" in evidence_html

    def test_chip_sits_at_the_offset_the_model_chose(self) -> None:
        report_html, _ = render_panels(investigation_with(report_with(citation()), [evidence()]))

        assert "Coverage was lost.<a class='cite'" in report_html

    def test_unverified_citation_is_flagged_and_highlights_nothing(self) -> None:
        report_html, evidence_html = render_panels(
            investigation_with(report_with(citation("E9", "L1", verified=False)), [evidence()])
        )

        assert "cite unverified" in report_html
        assert "⚠" in report_html
        assert "cited" not in evidence_html

    def test_statement_text_is_escaped(self) -> None:
        report = report_with(citation(), statement="Coverage was lost.")
        report = Report(
            summary=report.summary,
            findings=(Finding(statement="<script>alert(1)</script>", citations=()),),
            timeline=(),
            open_questions=(),
        )

        report_html, _ = render_panels(investigation_with(report, [evidence()]))

        assert "<script>" not in report_html
        assert "&lt;script&gt;" in report_html

    def test_reports_when_nothing_cleared_the_bar(self) -> None:
        report = Report(summary="Nothing found.", findings=(), timeline=(), open_questions=())

        _, evidence_html = render_panels(investigation_with(report, []))

        assert "No evidence met the relevance bar" in evidence_html

    def test_handles_an_investigation_with_no_report(self) -> None:
        report_html, evidence_html = render_panels(Investigation(question="What failed?"))

        assert "No report generated" in report_html
        assert evidence_html == ""


class TestMarkdown:
    def test_includes_citations_scores_and_numbered_lines(self) -> None:
        report = Report(
            summary="A camera lost coverage.",
            findings=(Finding(statement="Coverage was lost.", citations=(citation("E1", "L2"),)),),
            timeline=(),
            open_questions=("Who disabled it?",),
            excluded=(("E2", "separate matter"),),
        )

        markdown = report_markdown(investigation_with(report, [evidence()]))

        assert "**Question:** What failed?" in markdown
        assert "Coverage was lost. [E1·L2]" in markdown
        assert "relevance 0.87" in markdown
        assert "> L2: Engineers found the port had been disabled at the switch." in markdown
        assert "Retrieved but excluded" in markdown
        assert "Who disabled it?" in markdown
        assert "Citations: 100%" in markdown

    def test_flags_an_unverified_citation(self) -> None:
        report = Report(
            summary="",
            findings=(
                Finding(
                    statement="Coverage was lost.",
                    citations=(citation("E9", "L1", verified=False),),
                ),
            ),
            timeline=(),
            open_questions=(),
        )

        markdown = report_markdown(investigation_with(report, [evidence()]))

        assert "⚠" in markdown
        assert "Citations: 0%" in markdown

    def test_handles_an_investigation_with_no_report(self) -> None:
        assert "No report was generated" in report_markdown(Investigation(question="What?"))


class TestTrace:
    def test_reports_when_there_are_no_rounds(self) -> None:
        assert "No retrieval rounds" in retrieval_trace_markdown(Investigation(question="q"))
