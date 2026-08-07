"""Hybrid retrieval: dense vectors, Okapi BM25, and reciprocal rank fusion.

Both halves earn their place. Dense search generalises over paraphrase — a question about
"laundering" matches a passage that never uses the word. BM25 nails the literal tokens an
investigator actually types: proper nouns, addresses, IP addresses, product names, which
embeddings blur into their neighbourhoods. Fusing the two rank lists with RRF avoids
having to calibrate two incomparable score scales.
"""

from detective.retrieval.fusion import reciprocal_rank_fusion
from detective.retrieval.lexical import BM25Index
from detective.retrieval.tokens import tokenize
from detective.retrieval.vector import Vector, VectorIndex

__all__ = [
    "BM25Index",
    "Vector",
    "VectorIndex",
    "reciprocal_rank_fusion",
    "tokenize",
]
