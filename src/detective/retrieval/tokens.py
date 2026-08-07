"""Shared tokenisation.

One tokeniser for BM25, for the offline stand-in embedder and for the prompt-leakage test,
so all three agree on what counts as a word.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping dotted and hyphenated forms intact.

    Splitting ``192.168.0.1`` into four numbers would destroy exactly the identifiers a
    lexical index exists to match.
    """
    return _TOKEN.findall(text.lower())
