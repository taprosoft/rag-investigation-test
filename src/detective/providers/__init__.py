"""Model providers behind narrow protocols.

The pipeline depends only on the three protocols in :mod:`detective.providers.base`, so
the whole test suite runs offline against deterministic stand-ins and a provider swap
never reaches business logic. SDK ``Any`` is confined to the adapter modules.
"""

from detective.providers.base import ChatModel, Embedder, Reranker, parse_json_object
from detective.providers.cohere_provider import CohereReranker
from detective.providers.fakes import (
    HashEmbedder,
    LexicalOverlapReranker,
    ScriptedChatModel,
)
from detective.providers.openai_provider import OpenAIChatModel, OpenAIEmbedder

__all__ = [
    "ChatModel",
    "CohereReranker",
    "Embedder",
    "HashEmbedder",
    "LexicalOverlapReranker",
    "OpenAIChatModel",
    "OpenAIEmbedder",
    "Reranker",
    "ScriptedChatModel",
    "parse_json_object",
]
