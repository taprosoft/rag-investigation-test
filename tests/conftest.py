"""Shared fixtures. Everything here runs offline with deterministic stand-ins."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from detective.core.config import Settings
from detective.core.corpus import chunk_documents, load_documents
from detective.core.models import Chunk
from detective.investigation.pipeline import Investigator, Providers
from detective.providers import HashEmbedder, LexicalOverlapReranker, ScriptedChatModel
from detective.retrieval import BM25Index, VectorIndex

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's real ``.env`` and shell credentials out of the test run.

    ``Settings`` reads ``.env`` relative to the working directory, so running the suite
    from a scratch directory is enough to make it hermetic: otherwise whichever model
    names or keys happen to be configured locally would change test behaviour, and the
    suite would pass or fail depending on the machine.
    """
    monkeypatch.chdir(tmp_path)
    for name in (
        "OPENAI_API_KEY",
        "COHERE_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith("DETECTIVE_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at the synthetic corpus, with the cache isolated per test."""
    return Settings(
        corpus_dir=REPO_ROOT / "eval" / "synthetic",
        cache_dir=tmp_path / "cache",
        candidates=8,
        rerank_threshold=0.0,
        per_round_evidence=2,
        max_rounds=3,
        max_evidence=4,
    )


@pytest.fixture
def chunks(settings: Settings) -> list[Chunk]:
    return chunk_documents(load_documents(settings.corpus_dir))


def make_investigator(
    settings: Settings, responses: list[str], chat: ScriptedChatModel | None = None
) -> Investigator:
    """Build an investigator over the settings' corpus with a scripted chat model.

    Pass ``chat`` when a test needs to inspect the calls that were made; otherwise the
    ``responses`` list is enough.
    """
    documents = load_documents(settings.corpus_dir)
    corpus_chunks = chunk_documents(documents)
    embedder = HashEmbedder()
    vectors = VectorIndex(corpus_chunks, embedder.embed([c.text for c in corpus_chunks]))
    providers = Providers(
        embedder=embedder,
        reranker=LexicalOverlapReranker(),
        chat=chat or ScriptedChatModel(responses or ["{}"]),
    )
    return Investigator(
        documents, corpus_chunks, vectors, BM25Index(corpus_chunks), providers, settings
    )


def json_response(payload: object) -> str:
    return json.dumps(payload)
