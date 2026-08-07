"""Domain types, configuration and corpus loading — no I/O beyond the filesystem."""

from detective.core.config import Settings
from detective.core.corpus import (
    chunk_document,
    chunk_documents,
    corpus_fingerprint,
    load_documents,
)
from detective.core.models import (
    Chunk,
    Citation,
    Document,
    Evidence,
    Finding,
    Investigation,
    Report,
    RoundResult,
    ScoredChunk,
)

__all__ = [
    "Chunk",
    "Citation",
    "Document",
    "Evidence",
    "Finding",
    "Investigation",
    "Report",
    "RoundResult",
    "ScoredChunk",
    "Settings",
    "chunk_document",
    "chunk_documents",
    "corpus_fingerprint",
    "load_documents",
]
