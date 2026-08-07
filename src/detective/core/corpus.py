"""Loading and chunking a folder of plain-text case files."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from detective.core.models import Chunk, Document

_BLANK_LINE = re.compile(r"\n\s*\n")


def load_documents(corpus_dir: Path) -> list[Document]:
    """Read every ``*.txt`` under ``corpus_dir``, sorted by filename for stable ordering."""
    paths = sorted(corpus_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"no .txt files found in {corpus_dir}")
    return [
        Document(doc_id=p.stem, path=p, text=p.read_text(encoding="utf-8").strip()) for p in paths
    ]


def chunk_document(document: Document) -> list[Chunk]:
    """Split a document into paragraph chunks.

    Paragraphs are the natural unit here: each states one discrete fact about the case,
    so retrieving at paragraph level keeps citations tight while the ``doc_id`` still
    lets us present evidence per source file.
    """
    paragraphs = [p.strip() for p in _BLANK_LINE.split(document.text) if p.strip()]
    return [
        Chunk(chunk_id=f"{document.doc_id}#{i}", doc_id=document.doc_id, index=i, text=text)
        for i, text in enumerate(paragraphs)
    ]


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunk_document(document)]


def corpus_fingerprint(chunks: list[Chunk], model: str) -> str:
    """Content hash of the chunk set plus embedding model, used as the cache key."""
    digest = hashlib.sha256(model.encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(chunk.text.encode("utf-8"))
    return digest.hexdigest()[:16]
