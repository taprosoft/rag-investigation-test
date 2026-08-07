from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from detective.core.models import Chunk, ScoredChunk
from detective.providers import HashEmbedder
from detective.retrieval import (
    BM25Index,
    VectorIndex,
    reciprocal_rank_fusion,
    tokenize,
)


def chunk(chunk_id: str, text: str) -> Chunk:
    doc_id, _, index = chunk_id.partition("#")
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, index=int(index or 0), text=text)


def test_tokenizer_keeps_dotted_and_hyphenated_forms() -> None:
    assert tokenize("Login from 192.168.0.1 via well-known host") == [
        "login",
        "from",
        "192.168.0.1",
        "via",
        "well-known",
        "host",
    ]


class TestVectorIndex:
    def test_ranks_by_cosine_similarity(self) -> None:
        chunks = [chunk("a#0", "alpha"), chunk("b#0", "beta"), chunk("c#0", "gamma")]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
        index = VectorIndex(chunks, embeddings)

        results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)

        assert [r.chunk.chunk_id for r in results] == ["a#0", "c#0", "b#0"]
        assert results[0].score == pytest.approx(1.0)
        assert all(r.scorer == "dense" for r in results)

    def test_magnitude_does_not_affect_ranking(self) -> None:
        chunks = [chunk("a#0", "alpha"), chunk("b#0", "beta")]
        index = VectorIndex(chunks, np.array([[10.0, 0.0], [0.0, 0.1]], dtype=np.float32))

        results = index.search(np.array([0.01, 0.0], dtype=np.float32), top_k=1)

        assert results[0].chunk.chunk_id == "a#0"

    def test_rejects_mismatched_inputs(self) -> None:
        with pytest.raises(ValueError, match="1 chunks but 2 embeddings"):
            VectorIndex([chunk("a#0", "alpha")], np.zeros((2, 3), dtype=np.float32))

    def test_cache_roundtrip(self, tmp_path: Path) -> None:
        chunks = [chunk("a#0", "alpha beta"), chunk("b#0", "gamma delta")]
        embedder = HashEmbedder(dimensions=32)
        index = VectorIndex(chunks, embedder.embed([c.text for c in chunks]))
        path = tmp_path / "index.npz"
        index.save(path)

        restored = VectorIndex.load(path, chunks)

        assert restored is not None
        query = embedder.embed(["alpha beta"])[0]
        assert [r.chunk.chunk_id for r in restored.search(query, 2)] == [
            r.chunk.chunk_id for r in index.search(query, 2)
        ]

    def test_cache_is_ignored_when_the_corpus_changed(self, tmp_path: Path) -> None:
        chunks = [chunk("a#0", "alpha")]
        path = tmp_path / "index.npz"
        VectorIndex(chunks, np.ones((1, 4), dtype=np.float32)).save(path)

        assert VectorIndex.load(path, [chunk("z#0", "zeta")]) is None
        assert VectorIndex.load(tmp_path / "absent.npz", chunks) is None


class TestBM25:
    def test_prefers_the_document_containing_the_rare_term(self) -> None:
        chunks = [
            chunk("a#0", "the funds were moved through a mixer"),
            chunk("b#0", "the funds were moved to a wallet"),
            chunk("c#0", "the employee reported a phishing email"),
        ]

        results = BM25Index(chunks).search("mixer", top_k=3)

        assert results[0].chunk.chunk_id == "a#0"
        assert results[0].scorer == "lexical"

    def test_ignores_terms_absent_from_the_corpus(self) -> None:
        index = BM25Index([chunk("a#0", "alpha beta")])

        assert index.search("nonexistent", top_k=3) == []
        assert index.search("", top_k=3) == []

    def test_exact_identifiers_survive_tokenisation(self) -> None:
        chunks = [
            chunk("a#0", "connection from 203.0.113.7 was blocked"),
            chunk("b#0", "connection from an unfamiliar address was blocked"),
        ]

        results = BM25Index(chunks).search("203.0.113.7", top_k=2)

        assert [r.chunk.chunk_id for r in results] == ["a#0"]


class TestReciprocalRankFusion:
    def test_rewards_agreement_between_rankers(self) -> None:
        dense = [
            ScoredChunk(chunk("a#0", "a"), 0.9, "dense"),
            ScoredChunk(chunk("b#0", "b"), 0.8, "dense"),
        ]
        lexical = [
            ScoredChunk(chunk("b#0", "b"), 12.0, "lexical"),
            ScoredChunk(chunk("c#0", "c"), 11.0, "lexical"),
        ]

        fused = reciprocal_rank_fusion([dense, lexical], k=60)

        assert [f.chunk.chunk_id for f in fused] == ["b#0", "a#0", "c#0"]
        assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
        assert all(f.scorer == "fused" for f in fused)

    def test_incomparable_score_scales_do_not_dominate(self) -> None:
        """A huge BM25 score must not outweigh consensus; only rank position counts."""
        dense = [
            ScoredChunk(chunk("a#0", "a"), 0.51, "dense"),
            ScoredChunk(chunk("b#0", "b"), 0.50, "dense"),
        ]
        lexical = [ScoredChunk(chunk("b#0", "b"), 999.0, "lexical")]

        fused = reciprocal_rank_fusion([dense, lexical])

        assert fused[0].chunk.chunk_id == "b#0"

    def test_honours_top_k_and_empty_input(self) -> None:
        dense = [ScoredChunk(chunk(f"{i}#0", "x"), 1.0, "dense") for i in range(5)]

        assert len(reciprocal_rank_fusion([dense], top_k=2)) == 2
        assert reciprocal_rank_fusion([]) == []
