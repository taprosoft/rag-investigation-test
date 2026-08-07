"""Deterministic stand-ins used by the test suite and by ``--offline``.

None of these are good models, and none pretend to be. They exist so that retrieval,
fusion, the agent loop, citation parsing and the UI can all be exercised in CI with no
network, no credentials and no flakiness.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from detective.retrieval.tokens import tokenize
from detective.retrieval.vector import Vector


class HashEmbedder:
    """Bag-of-words hashing embedder: same text in, same vector out, always."""

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    @property
    def identity(self) -> str:
        return f"hash-stand-in:{self._dimensions}"

    def embed(self, texts: Sequence[str]) -> Vector:
        matrix = np.zeros((len(texts), self._dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                matrix[row, int.from_bytes(digest[:4], "big") % self._dimensions] += 1.0
        return matrix


class LexicalOverlapReranker:
    """Stand-in cross-encoder scoring by Jaccard overlap of tokens."""

    def rerank(self, query: str, passages: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        query_terms = set(tokenize(query))
        scored: list[tuple[int, float]] = []
        for index, passage in enumerate(passages):
            terms = set(tokenize(passage))
            union = query_terms | terms
            scored.append((index, len(query_terms & terms) / len(union) if union else 0.0))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_n]


class ScriptedChatModel:
    """Replays a fixed list of responses, then repeats the last one.

    Repeating rather than raising keeps a test focused on the rounds it cares about
    without having to script every downstream call.
    """

    def __init__(self, responses: Sequence[str]) -> None:
        if not responses:
            raise ValueError("ScriptedChatModel needs at least one response")
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, as_json: bool = False) -> str:
        self.calls.append((system, user))
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]
