from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from conftest import json_response, make_investigator
from detective.core.config import Settings
from detective.core.models import Chunk, Evidence
from detective.investigation.citations import marker
from detective.investigation.pipeline import Investigator, Providers, build_report
from detective.providers import HashEmbedder, LexicalOverlapReranker, ScriptedChatModel
from detective.retrieval import Vector

PASSAGE = (
    "The credential was presented three times. "
    "The door released on the third attempt. "
    "No visitor was booked in that night."
)


def evidence(label: str = "E1", text: str = PASSAGE) -> Evidence:
    return Evidence(
        label=label,
        chunk=Chunk(chunk_id="doc#0", doc_id="doc", index=0, text=text),
        rerank_score=0.9,
        round_index=1,
        query="q",
        rationale="because",
    )


class TestBuildReport:
    def test_separates_grounded_claims_from_ungrounded_ones(self) -> None:
        parsed: dict[str, object] = {
            "summary": f"A credential was misused.{marker('E1', 'L1')}",
            "findings": [
                f"The door eventually opened.{marker('E1', 'L2')}",
                f"The intruder wore a hat.{marker('E4', 'L1')}",
            ],
            "timeline": [f"A credential was presented repeatedly.{marker('E1', 'L1')}"],
            "open_questions": ["Who issued the credential?"],
            "excluded": [{"label": "E2", "reason": "separate matter"}],
        }

        report = build_report(parsed, [evidence()])

        assert report.summary == "A credential was misused."
        assert report.timeline[0].fully_grounded
        assert report.findings[0].fully_grounded
        assert not report.findings[1].fully_grounded
        assert report.grounding_rate == 2 / 3
        assert report.excluded == (("E2", "separate matter"),)
        assert report.open_questions == ("Who issued the credential?",)

    def test_an_uncited_claim_is_not_grounded(self) -> None:
        report = build_report({"findings": ["The door opened."]}, [evidence()])

        assert report.findings[0].citations == ()
        assert not report.findings[0].fully_grounded

    def test_survives_a_malformed_response(self) -> None:
        report = build_report({}, [evidence()])

        assert report.summary == ""
        assert report.findings == ()
        assert report.grounding_rate == 0.0

    def test_ignores_entries_of_the_wrong_shape(self) -> None:
        """A model that returns objects where strings were asked for must not crash us."""
        report = build_report({"findings": [{"statement": "x"}, "Real claim."]}, [evidence()])

        assert [f.statement for f in report.findings] == ["Real claim."]


class TestSingleStep:
    def test_admits_only_passages_above_the_threshold(self, settings: Settings) -> None:
        strict = settings.model_copy(update={"rerank_threshold": 0.9})
        investigator = make_investigator(strict, [json_response({})])

        investigation = investigator.investigate("How did the intruder get in?", mode="single")

        assert investigation.evidence == []
        assert investigation.rounds[0].rejected
        assert "below threshold" in investigation.rounds[0].rejected[0][2]
        assert investigation.report is not None
        assert "No passage" in investigation.report.summary

    def test_runs_exactly_one_round_and_labels_evidence(self, settings: Settings) -> None:
        investigator = make_investigator(settings, [json_response({"summary": "done"})])

        investigation = investigator.investigate("Where did the goods go?", mode="single")

        assert len(investigation.rounds) == 1
        assert investigation.rounds[0].sufficient
        assert [e.label for e in investigation.evidence] == ["E1", "E2"]
        assert len(investigation.evidence) <= settings.max_evidence
        assert all(e.round_index == 1 for e in investigation.evidence)

    def test_makes_a_single_llm_call(self, settings: Settings) -> None:
        """Single-step spends one model call, on synthesis. Otherwise it loses its point."""
        chat = ScriptedChatModel([json_response({"summary": "done"})])
        investigator = make_investigator(settings, [], chat=chat)

        investigator.investigate("Where did the goods go?", mode="single")

        assert len(chat.calls) == 1


class TestShortlistDiversity:
    def test_one_document_cannot_fill_the_shortlist(self, settings: Settings) -> None:
        """A verbose file must not crowd out the source holding the other half of an answer."""
        wide = settings.model_copy(update={"per_round_evidence": 6, "per_document_limit": 1})
        investigator = make_investigator(wide, [json_response({})])

        investigation = investigator.investigate("Which systems failed?", mode="single")

        doc_ids = [e.doc_id for e in investigation.evidence]
        assert len(doc_ids) == len(set(doc_ids))

    def test_limit_accounts_for_evidence_already_collected(self, settings: Settings) -> None:
        capped = settings.model_copy(
            update={"per_document_limit": 1, "max_rounds": 3, "max_evidence": 9}
        )
        investigator = make_investigator(
            capped,
            [json_response({"admit": [], "reject": [], "sufficient": False, "gap": "more"})],
        )

        investigation = investigator.investigate("Which systems failed?", mode="agentic")

        doc_ids = [e.doc_id for e in investigation.evidence]
        assert len(doc_ids) == len(set(doc_ids))


def top_candidate(
    investigator: Investigator, question: str, query: str | None = None, top_k: int = 8
) -> str:
    """The chunk id the pipeline actually surfaces first for ``query``.

    Mirrors the pipeline exactly — search on the round's query, rerank against the
    original question. Derived rather than hardcoded, because these tests are about the
    agent loop's control flow, not about how the deterministic stand-in providers happen
    to rank passages.
    """
    candidates = investigator.hybrid_search(query or question, top_k)
    return investigator.rerank(question, candidates)[0].chunk.chunk_id


class TestAgenticLoop:
    def test_stops_as_soon_as_the_assessor_is_satisfied(self, settings: Settings) -> None:
        question = "How did the intruder get in?"
        probe = make_investigator(settings, ["{}"])
        chunk_id = top_candidate(probe, question)
        investigator = make_investigator(
            settings,
            [
                json_response(
                    {
                        "admit": [{"id": chunk_id, "reason": "names the point of access"}],
                        "reject": [],
                        "sufficient": True,
                        "gap": "",
                    }
                ),
                json_response({"summary": "Access came from a stale contractor credential."}),
            ],
        )

        investigation = investigator.investigate(question, mode="agentic")

        assert len(investigation.rounds) == 1
        assert investigation.rounds[0].query == question
        assert investigation.rounds[0].reason.startswith("opening search")
        assert [e.chunk.chunk_id for e in investigation.evidence] == [chunk_id]
        assert investigation.evidence[0].rationale == "names the point of access"
        assert investigation.report is not None
        assert investigation.report.summary.startswith("Access came from")

    def test_keeps_searching_while_the_assessor_reports_a_gap(self, settings: Settings) -> None:
        question = "How did the intruder get in?"
        follow_up = "camera outage at the loading dock"
        probe = make_investigator(settings, ["{}"])
        chunk_id = top_candidate(probe, question, follow_up)
        investigator = make_investigator(
            settings,
            [
                # round 1: assessor admits nothing and names the gap
                json_response(
                    {"admit": [], "reject": [], "sufficient": False, "gap": "camera coverage"}
                ),
                # round 2: planner issues a targeted follow-up query
                json_response({"query": follow_up, "reason": "close the gap"}),
                # round 2: assessor admits what that query brought back, then stops
                json_response(
                    {
                        "admit": [{"id": chunk_id, "reason": "documents the coverage gap"}],
                        "sufficient": True,
                        "gap": "",
                    }
                ),
                json_response({"summary": "done"}),
            ],
        )

        investigation = investigator.investigate(question, mode="agentic")

        assert len(investigation.rounds) == 2
        assert investigation.rounds[0].admitted == ()
        assert investigation.rounds[0].gap == "camera coverage"
        assert investigation.rounds[1].query == follow_up
        assert investigation.rounds[1].reason == "close the gap"
        assert [e.chunk.chunk_id for e in investigation.evidence] == [chunk_id]

    def test_falls_back_when_the_planner_repeats_a_query(self, settings: Settings) -> None:
        """A planner that loops on one query would stall the search; widen instead."""
        question = "How did the intruder get in?"
        investigator = make_investigator(
            settings,
            [
                json_response(
                    {"admit": [], "reject": [], "sufficient": False, "gap": "camera coverage"}
                ),
                json_response({"query": question, "reason": "same again"}),
                json_response({"admit": [], "sufficient": True, "gap": ""}),
            ],
        )

        investigation = investigator.investigate(question, mode="agentic")

        assert investigation.rounds[1].query == f"{question} camera coverage"
        assert "widened" in investigation.rounds[1].reason

    def test_respects_the_round_ceiling(self, settings: Settings) -> None:
        capped = settings.model_copy(update={"max_rounds": 2, "max_evidence": 99})
        investigator = make_investigator(
            capped,
            [json_response({"admit": [], "reject": [], "sufficient": False, "gap": "more"})],
        )

        investigation = investigator.investigate("What happened?", mode="agentic")

        assert len(investigation.rounds) <= 2

    def test_never_admits_the_same_passage_twice(self, settings: Settings) -> None:
        investigator = make_investigator(
            settings,
            [
                json_response(
                    {
                        "admit": [{"id": "syn_02_camera_outage#0", "reason": "relevant"}],
                        "sufficient": False,
                        "gap": "more detail",
                    }
                )
            ],
        )

        investigation = investigator.investigate("Which systems failed?", mode="agentic")

        ids = [e.chunk.chunk_id for e in investigation.evidence]
        assert len(ids) == len(set(ids))
        assert [e.label for e in investigation.evidence] == [
            f"E{i}" for i in range(1, len(ids) + 1)
        ]


class CountingEmbedder(HashEmbedder):
    """Counts embedding calls so the cache can be observed rather than assumed."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> Vector:
        self.calls += 1
        return super().embed(texts)


def test_index_is_cached_between_builds(settings: Settings, tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    providers = Providers(
        embedder=embedder,
        reranker=LexicalOverlapReranker(),
        chat=ScriptedChatModel(["{}"]),
    )
    cached = settings.model_copy(update={"cache_dir": tmp_path / "cache"})

    Investigator.build(providers, cached)
    Investigator.build(providers, cached)

    assert embedder.calls == 1, "second build should reuse the cached vectors"


def test_cache_can_be_bypassed(settings: Settings, tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    providers = Providers(
        embedder=embedder, reranker=LexicalOverlapReranker(), chat=ScriptedChatModel(["{}"])
    )
    cached = settings.model_copy(update={"cache_dir": tmp_path / "cache"})

    Investigator.build(providers, cached, use_cache=False)
    Investigator.build(providers, cached, use_cache=False)

    assert embedder.calls == 2
