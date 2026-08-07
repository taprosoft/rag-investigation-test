from __future__ import annotations

from pathlib import Path

from conftest import json_response, make_investigator
from detective.core.config import Settings
from detective.core.models import Chunk, ScoredChunk
from detective.evaluation import (
    GoldenQuery,
    documents_in_order,
    evaluate_all,
    format_table,
    load_goldens,
    score_query,
)


def scored(doc_id: str, index: int = 0) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=f"{doc_id}#{index}", doc_id=doc_id, index=index, text="x"),
        score=1.0,
        scorer="fused",
    )


class TestMetrics:
    def test_collapses_chunks_to_documents_keeping_first_appearance(self) -> None:
        results = [scored("a", 0), scored("a", 1), scored("b"), scored("c")]

        assert documents_in_order(results, limit=2) == ["a", "b"]

    def test_scores_a_perfect_ranking(self) -> None:
        golden = GoldenQuery("q", frozenset({"a", "b"}), frozenset({"z"}))

        precision, recall, mrr, distractors = score_query(["a", "b"], golden)

        assert (precision, recall, mrr, distractors) == (1.0, 1.0, 1.0, 0.0)

    def test_penalises_a_distractor_and_a_late_first_hit(self) -> None:
        golden = GoldenQuery("q", frozenset({"a"}), frozenset({"z"}))

        precision, recall, mrr, distractors = score_query(["z", "a"], golden)

        # One relevant document, so R-precision looks only at rank 1 — which is a
        # distractor. Recall is still perfect: the right document was found, just late.
        assert precision == 0.0
        assert recall == 1.0
        assert mrr == 0.5
        assert distractors == 0.5

    def test_r_precision_is_not_capped_by_the_result_count(self) -> None:
        """precision@3 would cap a single-answer question at 0.33 however well it ranked."""
        golden = GoldenQuery("q", frozenset({"a"}), frozenset())

        precision, _, _, _ = score_query(["a", "b", "c"], golden)

        assert precision == 1.0

    def test_empty_retrieval_scores_zero(self) -> None:
        golden = GoldenQuery("q", frozenset({"a"}), frozenset())

        assert score_query([], golden) == (0.0, 0.0, 0.0, 0.0)


class TestGoldenSet:
    def test_loads_the_synthetic_set(self, repo_root: Path) -> None:
        goldens = load_goldens(repo_root / "eval" / "goldens.json")

        assert goldens.corpus_dir == repo_root / "eval" / "synthetic"
        assert len(goldens.queries) >= 5
        assert all(q.relevant for q in goldens.queries)

    def test_every_labelled_document_exists(self, repo_root: Path) -> None:
        goldens = load_goldens(repo_root / "eval" / "goldens.json")
        available = {p.stem for p in goldens.corpus_dir.glob("*.txt")}

        for query in goldens.queries:
            assert query.relevant <= available, query.question
            assert query.distractors <= available, query.question

    def test_relevant_and_distractor_sets_never_overlap(self, repo_root: Path) -> None:
        goldens = load_goldens(repo_root / "eval" / "goldens.json")

        for query in goldens.queries:
            assert not (query.relevant & query.distractors), query.question


class TestEndToEnd:
    def test_scores_every_strategy_on_the_synthetic_corpus(
        self, repo_root: Path, settings: Settings
    ) -> None:
        goldens = load_goldens(repo_root / "eval" / "goldens.json")
        investigator = make_investigator(settings, [json_response({})])

        results = evaluate_all(investigator, goldens, top_k=3)

        assert [m.strategy for m in results] == ["dense", "lexical", "hybrid", "hybrid+rerank"]
        assert all(0.0 <= m.precision <= 1.0 for m in results)
        assert all(0.0 <= m.recall <= 1.0 for m in results)

    def test_fusion_never_loses_a_document_either_half_found(
        self, repo_root: Path, settings: Settings
    ) -> None:
        """The structural guarantee fusion actually offers: coverage is the union.

        Not asserted here: that hybrid *outranks* both halves. The measured result is that
        dense alone can outrank the fusion on this data, and a test asserting otherwise
        would be marketing rather than a check — REPORT.md carries the numbers.
        """
        goldens = load_goldens(repo_root / "eval" / "goldens.json")
        investigator = make_investigator(settings, [json_response({})])
        pool = 40

        for query in goldens.queries:
            embedding = investigator.embedder.embed([query.question])[0]
            dense = {s.chunk.chunk_id for s in investigator.vector_index.search(embedding, pool)}
            lexical = {
                s.chunk.chunk_id for s in investigator.lexical_index.search(query.question, pool)
            }
            hybrid = {s.chunk.chunk_id for s in investigator.hybrid_search(query.question, pool)}

            assert dense | lexical == hybrid, query.question

    def test_table_renders_one_row_per_strategy(self, repo_root: Path, settings: Settings) -> None:
        goldens = load_goldens(repo_root / "eval" / "goldens.json")
        investigator = make_investigator(settings, [json_response({})])

        table = format_table(evaluate_all(investigator, goldens))

        assert table.count("\n") == 5
        assert "| hybrid+rerank |" in table
