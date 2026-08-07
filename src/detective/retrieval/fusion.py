"""Reciprocal rank fusion."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from detective.core.models import Chunk, ScoredChunk


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[ScoredChunk]],
    k: int = 60,
    top_k: int | None = None,
    weights: Sequence[float] | None = None,
) -> list[ScoredChunk]:
    """Fuse ranked lists by ``sum(weight / (k + rank))``.

    Rank-based fusion sidesteps the fact that a cosine similarity and a BM25 score live on
    entirely different scales — normalising them against each other would need a
    calibration set we do not have. ``k=60`` is the value from the original RRF paper; it
    damps the influence of any single list's top hit, so a passage both rankers like beats
    a passage one ranker loves.

    ``weights`` exists because equal weighting silently assumes the rankers are equally
    good. When one is clearly stronger on a corpus — as measured, not as assumed — equal
    weighting lets the weaker list drag the stronger one down. See ``eval/`` for the
    measurement behind the default.
    """
    lists = list(rankings)
    factors = list(weights) if weights is not None else [1.0] * len(lists)
    if len(factors) != len(lists):
        raise ValueError(f"got {len(lists)} rankings but {len(factors)} weights")

    totals: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking, weight in zip(lists, factors, strict=True):
        for rank, scored in enumerate(ranking, start=1):
            chunk_id = scored.chunk.chunk_id
            totals[chunk_id] = totals.get(chunk_id, 0.0) + weight / (k + rank)
            by_id[chunk_id] = scored.chunk
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [ScoredChunk(chunk=by_id[cid], score=score, scorer="fused") for cid, score in ordered]
