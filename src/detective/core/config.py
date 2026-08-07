"""Runtime configuration, sourced entirely from the environment / `.env`.

Nothing here is specific to a particular case: point ``corpus_dir`` at any folder of
``.txt`` files and the whole system works unchanged.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings.

    Tunables are namespaced under ``DETECTIVE_``; provider credentials keep their
    conventional bare names so an already-configured shell just works. Keys are optional
    so the test suite, the fake providers and ``--help`` run on a machine with no
    credentials at all.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DETECTIVE_",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: str = Field(default="", validation_alias=AliasChoices("OPENAI_API_KEY"))
    cohere_api_key: str = Field(default="", validation_alias=AliasChoices("COHERE_API_KEY"))

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    rerank_model: str = "rerank-v3.5"

    corpus_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")

    #: Hybrid candidates pulled per search round before reranking.
    candidates: int = 15
    #: Weight given to the dense ranker relative to BM25 when fusing. Measured, not
    #: assumed: see REPORT.md. Set to 1.0 for textbook equal-weight RRF.
    dense_weight: float = 2.0
    #: Reranker score below which a passage is never admitted as evidence.
    rerank_threshold: float = 0.30
    #: Passages put in front of the gatekeeper per round. The gatekeeper, not this number,
    #: is the precision control — cutting the shortlist too early hides evidence from the
    #: only stage able to judge it.
    per_round_evidence: int = 5
    #: Passages one source file may contribute to a single shortlist. Without a cap, a
    #: verbose document crowds out every other source and the answer narrows to whatever
    #: one file happens to say.
    per_document_limit: int = 2
    #: Hard ceiling on agent search rounds, so a bad plan cannot loop forever.
    max_rounds: int = 4
    #: Hard ceiling on the final evidence set.
    max_evidence: int = 6

    s3_bucket: str = ""
    aws_region: str = Field(default="ap-southeast-1", validation_alias=AliasChoices("AWS_REGION"))
    aws_access_key_id: str = Field(default="", validation_alias=AliasChoices("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key: str = Field(
        default="", validation_alias=AliasChoices("AWS_SECRET_ACCESS_KEY")
    )

    @property
    def live_providers_available(self) -> bool:
        """True when both the chat/embedding key and the rerank key are configured."""
        return bool(self.openai_api_key and self.cohere_api_key)

    @property
    def s3_configured(self) -> bool:
        return bool(self.s3_bucket and self.aws_access_key_id and self.aws_secret_access_key)
