"""Report archival."""

from detective.storage.s3 import build_key, slugify, upload_report

__all__ = ["build_key", "slugify", "upload_report"]
