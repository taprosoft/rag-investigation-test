"""Citation marker handling — the contract between the synthesiser and the UI."""

from __future__ import annotations

import pytest

from detective.core.models import Chunk, Evidence
from detective.investigation.citations import (
    CITATION_DELIMITER,
    CITATION_START,
    CITATION_STOP,
    extract_citations,
    format_evidence,
    marker,
    parse_claim,
    parse_locator,
    strip_citations,
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


class TestSentenceLines:
    def test_passage_splits_into_addressable_lines(self) -> None:
        assert len(evidence().lines) == 3
        assert evidence().lines[1].startswith("Engineers found")

    def test_line_text_resolves_a_range(self) -> None:
        assert evidence().line_text((1, 2)).startswith("Camera 4 stopped")
        assert evidence().line_text((99,)) == ""


class TestExtract:
    def test_extracts_label_locator_and_offsets(self) -> None:
        text = f"Coverage was lost.{marker('E1', 'L1')} Then it returned."

        found = extract_citations(text)

        assert len(found) == 1
        assert found[0].source_ids == ("E1",)
        assert found[0].locator == "L1"
        assert text[found[0].start : found[0].end] == marker("E1", "L1")

    def test_accepts_a_marker_without_a_locator(self) -> None:
        found = extract_citations(f"Claim.{marker('E2')}")

        assert found[0].source_ids == ("E2",)
        assert found[0].locator is None

    def test_reads_a_multi_source_marker(self) -> None:
        raw = (
            f"{CITATION_START}cite{CITATION_DELIMITER}E1"
            f"{CITATION_DELIMITER}E2{CITATION_DELIMITER}L2{CITATION_STOP}"
        )

        found = extract_citations(f"Claim.{raw}")

        assert found[0].source_ids == ("E1", "E2")
        assert found[0].locator == "L2"

    @pytest.mark.parametrize(
        "text",
        [
            "No markers at all.",
            f"Truncated {CITATION_START}cite{CITATION_DELIMITER}E1",
            f"{CITATION_START}cite{CITATION_DELIMITER}{CITATION_STOP}",
            f"{CITATION_START}cite{CITATION_DELIMITER}bad id!{CITATION_STOP}",
        ],
    )
    def test_malformed_markers_are_dropped_not_raised(self, text: str) -> None:
        """Losing one citation beats losing the whole report to a generation artefact."""
        assert extract_citations(text) == []

    def test_strip_removes_every_marker(self) -> None:
        text = f"One.{marker('E1', 'L1')} Two.{marker('E2', 'L3')}"

        assert strip_citations(text, extract_citations(text)) == "One. Two."


class TestParseLocator:
    @pytest.mark.parametrize(
        ("locator", "expected"),
        [
            ("L1", (1,)),
            ("L2-L3", (2, 3)),
            ("L3-L2", (2, 3)),
            ("L2-L9", (2, 3)),
            (None, (1, 2, 3)),
        ],
    )
    def test_resolves_within_the_passage(
        self, locator: str | None, expected: tuple[int, ...]
    ) -> None:
        assert parse_locator(locator, line_count=3) == expected

    @pytest.mark.parametrize("locator", ["L9", "L0", "page 3", "LL1"])
    def test_unresolvable_locators_return_nothing(self, locator: str) -> None:
        assert parse_locator(locator, line_count=3) == ()


class TestParseClaim:
    def test_returns_clean_prose_with_verified_citations(self) -> None:
        text = f"The camera lost coverage at 02:03.{marker('E1', 'L1')}"

        statement, citations = parse_claim(text, [evidence()])

        assert statement == "The camera lost coverage at 02:03."
        assert len(citations) == 1
        assert citations[0].verified
        assert citations[0].lines == (1,)
        assert citations[0].display == "E1·L1"

    def test_display_normalises_the_locator_to_what_resolved(self) -> None:
        """``L1-L1`` and an over-long range are the model's noise, not the reader's."""
        _, single = parse_claim(f"Claim.{marker('E1', 'L1-L1')}", [evidence()])
        _, ranged = parse_claim(f"Claim.{marker('E1', 'L1-L9')}", [evidence()])
        _, unknown = parse_claim(f"Claim.{marker('E9', 'L1')}", [evidence()])

        assert single[0].display == "E1·L1"
        assert ranged[0].display == "E1·L1-L3"
        assert unknown[0].display == "E9"

    def test_offsets_track_the_cleaned_text(self) -> None:
        text = f"First.{marker('E1', 'L1')} Second.{marker('E1', 'L3')}"

        statement, citations = parse_claim(text, [evidence()])

        assert statement == "First. Second."
        assert [c.offset for c in citations] == [6, 14]
        assert statement[: citations[0].offset] == "First."

    def test_unknown_label_is_kept_but_unverified(self) -> None:
        statement, citations = parse_claim(f"Claim.{marker('E9', 'L1')}", [evidence()])

        assert statement == "Claim."
        assert citations[0].verified is False
        assert citations[0].lines == ()

    def test_out_of_range_line_is_unverified(self) -> None:
        _, citations = parse_claim(f"Claim.{marker('E1', 'L9')}", [evidence()])

        assert citations[0].verified is False

    def test_a_multi_source_marker_becomes_one_citation_per_source(self) -> None:
        raw = (
            f"{CITATION_START}cite{CITATION_DELIMITER}E1"
            f"{CITATION_DELIMITER}E2{CITATION_DELIMITER}L1{CITATION_STOP}"
        )

        _, citations = parse_claim(f"Claim.{raw}", [evidence("E1"), evidence("E2")])

        assert [c.label for c in citations] == ["E1", "E2"]
        assert all(c.verified for c in citations)


class TestFormatEvidence:
    def test_numbers_lines_for_the_synthesiser(self) -> None:
        rendered = format_evidence([evidence()])

        assert "[E1] source: doc" in rendered
        assert "L1: Camera 4 stopped recording at 02:03." in rendered
        assert "L3: No other camera lost coverage." in rendered

    def test_unnumbered_form_is_used_where_locators_are_irrelevant(self) -> None:
        rendered = format_evidence([evidence()], numbered=False)

        assert "L1:" not in rendered
        assert PASSAGE in rendered

    def test_empty_evidence_renders_empty(self) -> None:
        assert format_evidence([]) == ""
