"""In-memory dense index over L2-normalised embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from detective.core.models import Chunk, ScoredChunk

Vector = NDArray[np.float32]


class VectorIndex:
    """Brute-force cosine similarity over a single matrix.

    A dedicated vector database buys nothing at this scale — the whole corpus is one small
    matrix and exhaustive cosine is exact and instant. The interface is narrow enough that
    swapping in Chroma or Qdrant later is a one-class change.
    """

    def __init__(self, chunks: Sequence[Chunk], embeddings: Vector) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"got {len(chunks)} chunks but {embeddings.shape[0]} embeddings")
        self._chunks = list(chunks)
        self._matrix = l2_normalise(embeddings.astype(np.float32))

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def search(self, query_embedding: Vector, top_k: int) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        query = l2_normalise(query_embedding.astype(np.float32).reshape(1, -1))[0]
        scores = self._matrix @ query
        order = np.argsort(-scores)[:top_k]
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]), scorer="dense")
            for i in order
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, matrix=self._matrix, chunk_ids=np.array([c.chunk_id for c in self._chunks])
        )

    @classmethod
    def load(cls, path: Path, chunks: Sequence[Chunk]) -> VectorIndex | None:
        """Reload cached vectors, or ``None`` if the cache does not match ``chunks``.

        Returning ``None`` rather than raising lets the caller treat a stale cache as a
        cache miss, which is what it is.
        """
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            cached_ids = [str(x) for x in data["chunk_ids"]]
            if cached_ids != [c.chunk_id for c in chunks]:
                return None
            return cls(chunks, data["matrix"].astype(np.float32))


def l2_normalise(matrix: Vector) -> Vector:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    normalised: Vector = (matrix / np.maximum(norms, 1e-12)).astype(np.float32)
    return normalised
