from __future__ import annotations

from pathlib import Path

import pytest

from detective.core.corpus import (
    chunk_document,
    chunk_documents,
    corpus_fingerprint,
    load_documents,
)
from detective.core.models import Document


def test_loads_every_case_file_in_stable_order(repo_root: Path) -> None:
    documents = load_documents(repo_root / "data")

    assert [d.doc_id for d in documents] == [f"case_{i}" for i in range(1, 9)]
    assert all(d.text for d in documents)


def test_missing_corpus_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)


def test_chunks_on_blank_lines_with_stable_ids() -> None:
    document = Document(doc_id="doc", path=Path("doc.txt"), text="first para\n\n  \n\nsecond para")

    chunks = chunk_document(document)

    assert [c.chunk_id for c in chunks] == ["doc#0", "doc#1"]
    assert [c.text for c in chunks] == ["first para", "second para"]
    assert all(c.doc_id == "doc" for c in chunks)


def test_fingerprint_tracks_content_and_model(repo_root: Path) -> None:
    chunks = chunk_documents(load_documents(repo_root / "data"))
    baseline = corpus_fingerprint(chunks, "model-a")

    assert corpus_fingerprint(chunks, "model-a") == baseline
    assert corpus_fingerprint(chunks, "model-b") != baseline
    assert corpus_fingerprint(chunks[:-1], "model-a") != baseline
