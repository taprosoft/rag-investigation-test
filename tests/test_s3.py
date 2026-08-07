from __future__ import annotations

from datetime import UTC, datetime

from detective.core.config import Settings
from detective.storage.s3 import build_key, slugify, upload_report

WHEN = datetime(2026, 8, 7, 9, 30, 0, tzinfo=UTC)


class StubStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> object:
        self.calls.append({"Bucket": Bucket, "Key": Key, "Body": Body, "ContentType": ContentType})
        return {}


def test_slugify_is_safe_and_bounded() -> None:
    assert (
        slugify("How did the hacker launder the funds?") == "how-did-the-hacker-launder-the-funds"
    )
    assert slugify("!!!") == "report"
    assert len(slugify("word " * 50)) <= 60


def test_key_is_sortable_and_descriptive() -> None:
    assert build_key("Where did the funds go?", WHEN) == (
        "reports/20260807T093000Z-where-did-the-funds-go.md"
    )


def test_uploads_markdown_and_returns_the_uri() -> None:
    settings = Settings(s3_bucket="mbg-devops-test")
    store = StubStore()

    uri = upload_report("# report", "Where did the funds go?", settings, client=store, when=WHEN)

    assert uri == "s3://mbg-devops-test/reports/20260807T093000Z-where-did-the-funds-go.md"
    assert store.calls[0]["Body"] == b"# report"
    assert store.calls[0]["ContentType"] == "text/markdown; charset=utf-8"


def test_is_a_no_op_when_unconfigured() -> None:
    assert upload_report("# report", "q", Settings()) is None


def test_configuration_requires_bucket_and_both_credentials() -> None:
    assert not Settings(s3_bucket="b").s3_configured
    assert not Settings(s3_bucket="b", aws_access_key_id="k").s3_configured
    assert Settings(s3_bucket="b", aws_access_key_id="k", aws_secret_access_key="s").s3_configured
