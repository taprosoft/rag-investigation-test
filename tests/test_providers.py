from __future__ import annotations

import pytest

from detective.providers import (
    HashEmbedder,
    LexicalOverlapReranker,
    ScriptedChatModel,
    parse_json_object,
)


class TestParseJsonObject:
    def test_parses_a_plain_object(self) -> None:
        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_recovers_from_a_code_fence(self) -> None:
        assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_recovers_an_object_wrapped_in_prose(self) -> None:
        assert parse_json_object('Sure! {"a": 1} Hope that helps.') == {"a": 1}

    @pytest.mark.parametrize("raw", ["", "not json", "[1, 2, 3]", "{oops"])
    def test_returns_an_empty_mapping_rather_than_raising(self, raw: str) -> None:
        assert parse_json_object(raw) == {}


class TestHashEmbedder:
    def test_is_deterministic_and_shaped_correctly(self) -> None:
        embedder = HashEmbedder(dimensions=64)

        first = embedder.embed(["alpha beta", "gamma"])
        second = embedder.embed(["alpha beta", "gamma"])

        assert first.shape == (2, 64)
        assert (first == second).all()

    def test_shared_vocabulary_produces_closer_vectors(self) -> None:
        embedder = HashEmbedder(dimensions=256)

        vectors = embedder.embed(
            ["the funds were moved through a mixer", "the funds were moved", "unrelated text here"]
        )
        similar = float(vectors[0] @ vectors[1])
        different = float(vectors[0] @ vectors[2])

        assert similar > different


class TestLexicalOverlapReranker:
    def test_orders_by_overlap_and_honours_top_n(self) -> None:
        reranker = LexicalOverlapReranker()

        results = reranker.rerank(
            "camera outage", ["nothing relevant", "the camera outage was logged"], top_n=1
        )

        assert results[0][0] == 1
        assert 0.0 < results[0][1] <= 1.0

    def test_handles_an_empty_candidate_list(self) -> None:
        assert LexicalOverlapReranker().rerank("q", [], top_n=3) == []


class TestScriptedChatModel:
    def test_replays_then_repeats_the_final_response(self) -> None:
        chat = ScriptedChatModel(["one", "two"])

        assert [chat.complete("s", "u") for _ in range(3)] == ["one", "two", "two"]
        assert len(chat.calls) == 3

    def test_requires_at_least_one_response(self) -> None:
        with pytest.raises(ValueError, match="at least one response"):
            ScriptedChatModel([])
