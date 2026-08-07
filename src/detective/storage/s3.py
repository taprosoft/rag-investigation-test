"""Optional report archival to S3.

Disabled unless credentials and a bucket are configured, so the rest of the system runs
unchanged on a machine with no AWS access.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from detective.core.config import Settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")


class ObjectStore(Protocol):
    """The one S3 operation we need, so tests can pass a stub."""

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> object: ...


def slugify(text: str, max_length: int = 60) -> str:
    slug = _SLUG.sub("-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "report"


def build_key(question: str, when: datetime) -> str:
    return f"reports/{when.strftime('%Y%m%dT%H%M%SZ')}-{slugify(question)}.md"


def upload_report(
    markdown: str,
    question: str,
    settings: Settings,
    *,
    client: ObjectStore | None = None,
    when: datetime | None = None,
) -> str | None:
    """Upload a report and return its ``s3://`` URI, or ``None`` when not configured."""
    if client is None:
        if not settings.s3_configured:
            logger.info("S3 upload skipped: bucket or credentials not configured")
            return None
        client = _default_client(settings)

    key = build_key(question, when or datetime.now(UTC))
    client.put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=markdown.encode("utf-8"),
        ContentType="text/markdown; charset=utf-8",
    )
    return f"s3://{settings.s3_bucket}/{key}"


def _default_client(settings: Settings) -> S3Client:
    import boto3

    client: S3Client = boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
    return client
