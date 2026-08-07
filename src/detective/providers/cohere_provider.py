"""Cohere rerank adapter."""

from __future__ import annotations

from collections.abc import Sequence


class CohereReranker:
    """Cohere rerank.

    This is the stage that separates *topically similar* from *actually relevant*. A
    bi-encoder scores query and passage independently, so a passage that discusses the
    right subject while explicitly ruling itself out still lands close in vector space. A
    cross-encoder reads the pair together and can act on that distinction, which is exactly
    the failure mode a corpus seeded with plausible distractors is built to expose.
    """

    def __init__(self, api_key: str, model: str) -> None:
        import cohere

        self._client = cohere.ClientV2(api_key=api_key)
        self._model = model

    def rerank(self, query: str, passages: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        if not passages:
            return []
        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=list(passages),
            top_n=min(top_n, len(passages)),
        )
        return [(int(r.index), float(r.relevance_score)) for r in response.results]
