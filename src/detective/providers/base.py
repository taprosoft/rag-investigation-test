"""Provider protocols and shared response parsing."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol, cast

from detective.retrieval.vector import Vector


class Embedder(Protocol):
    """Turns text into vectors. Implementations should batch."""

    @property
    def identity(self) -> str:
        """Stable id of *this* embedder, used to key the vector cache.

        The configured model name is not enough: swapping in the offline stand-in leaves
        the name untouched while producing entirely different vectors, so a cache keyed on
        the name alone will happily serve 256-dimensional stand-in vectors to a live
        1536-dimensional run.
        """

    def embed(self, texts: Sequence[str]) -> Vector: ...


class Reranker(Protocol):
    """Cross-encoder relevance scoring of passages against a query."""

    def rerank(self, query: str, passages: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        """Return ``(passage_index, score)`` pairs, best first, scores in ``[0, 1]``."""


class ChatModel(Protocol):
    """Instruction-following text generation, optionally constrained to JSON."""

    def complete(self, system: str, user: str, *, as_json: bool = False) -> str: ...


def parse_json_object(raw: str) -> dict[str, object]:
    """Best-effort parse of a model's JSON reply.

    Even in JSON mode a model can wrap output in a code fence; recovering the outermost
    object is cheaper than a retry, and the caller validates the shape either way. Returns
    an empty mapping rather than raising, so one malformed reply degrades a single stage
    instead of failing the whole investigation.
    """
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return cast("dict[str, object]", parsed) if isinstance(parsed, dict) else {}
