"""Okapi BM25 over the same chunks as the dense index."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from detective.core.models import Chunk, ScoredChunk
from detective.retrieval.tokens import tokenize


class BM25Index:
    """Classic Okapi BM25.

    Fifty lines is cheaper than a dependency, and having the implementation in-tree makes
    the ranking unit-testable rather than a black box.
    """

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b
        self._docs = [Counter(tokenize(c.text)) for c in self._chunks]
        self._lengths = np.array([sum(d.values()) for d in self._docs], dtype=np.float32)
        self._avg_length = float(self._lengths.mean()) if len(self._lengths) else 0.0
        total = len(self._chunks)
        frequencies = Counter(term for doc in self._docs for term in doc)
        self._idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5)) for term, df in frequencies.items()
        }

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms or not self._chunks:
            return []
        scores = np.zeros(len(self._chunks), dtype=np.float32)
        for term in terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, doc in enumerate(self._docs):
                tf = doc.get(term, 0)
                if not tf:
                    continue
                norm = 1 - self._b + self._b * self._lengths[i] / max(self._avg_length, 1e-9)
                scores[i] += idf * tf * (self._k1 + 1) / (tf + self._k1 * norm)
        order = [i for i in np.argsort(-scores) if scores[i] > 0][:top_k]
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]), scorer="lexical")
            for i in order
        ]
