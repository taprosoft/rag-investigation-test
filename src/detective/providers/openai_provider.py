"""OpenAI adapters for embeddings and chat."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from detective.retrieval.vector import Vector


class OpenAIEmbedder:
    """OpenAI embeddings. Batches the whole corpus into a single request."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def identity(self) -> str:
        return f"openai:{self._model}"

    def embed(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        return np.array([item.embedding for item in ordered], dtype=np.float32)


class OpenAIChatModel:
    """OpenAI chat completions with an optional JSON-object response format.

    Temperature is pinned to zero: the planner, the gatekeeper and the synthesiser are all
    doing judgement under a fixed rubric, where run-to-run variation is noise rather than
    creativity, and it makes the pipeline far easier to debug.
    """

    def __init__(self, api_key: str, model: str, temperature: float = 0.0) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def complete(self, system: str, user: str, *, as_json: bool = False) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            response_format={"type": "json_object"} if as_json else {"type": "text"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
