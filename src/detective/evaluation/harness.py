"""Retrieval evaluation.

"Hybrid plus reranking is better" is a claim, not a result. This module turns it into
numbers against a labelled set, and CI publishes the table on every run so a regression in
retrieval quality is as visible as a failing test.

Deliberately measured at *document* level: a detective cares which case files to read, not
which paragraph scored highest.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from detective.core.models import ScoredChunk
from detective.investigation.pipeline import Investigator, Mode


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    """One labelled question: what should come back, and what must not."""

    question: str
    relevant: frozenset[str]
    distractors: frozenset[str]


@dataclass(frozen=True, slots=True)
class GoldenSet:
    corpus_dir: Path
    queries: tuple[GoldenQuery, ...]


@dataclass(frozen=True, slots=True)
class Metrics:
    """Document-level retrieval quality for one strategy over the whole golden set."""

    strategy: str
    precision: float
    recall: float
    mrr: float
    distractor_rate: float

    def as_row(self) -> str:
        return (
            f"| {self.strategy} | {self.precision:.2f} | {self.recall:.2f} "
            f"| {self.mrr:.2f} | {self.distractor_rate:.2f} |"
        )


HEADER = (
    "| strategy | R-precision | recall@k | MRR | distractor rate |\n| --- | --- | --- | --- | --- |"
)


def load_goldens(path: Path) -> GoldenSet:
    data = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    return GoldenSet(
        corpus_dir=root / str(data["corpus"]),
        queries=tuple(
            GoldenQuery(
                question=str(q["question"]),
                relevant=frozenset(str(d) for d in q["relevant"]),
                distractors=frozenset(str(d) for d in q.get("distractors", [])),
            )
            for q in data["queries"]
        ),
    )


def documents_in_order(results: Sequence[ScoredChunk], limit: int) -> list[str]:
    """Collapse a chunk ranking to a document ranking, keeping first appearance."""
    ordered: dict[str, None] = {}
    for scored in results:
        ordered.setdefault(scored.chunk.doc_id, None)
        if len(ordered) == limit:
            break
    return list(ordered)


def score_query(retrieved: Sequence[str], golden: GoldenQuery) -> tuple[float, float, float, float]:
    """R-precision, recall, reciprocal rank and distractor rate for one question.

    Precision is measured at ``R = |relevant|`` rather than at a fixed ``k``. With a fixed
    cut-off, a question with one right answer can never score above ``1/k`` however
    perfectly it is answered, so plain precision@k would mostly measure how many relevant
    documents each question happens to have. R-precision removes that artefact and stays
    comparable across questions.
    """
    if not retrieved:
        return 0.0, 0.0, 0.0, 0.0
    hits = [doc for doc in retrieved if doc in golden.relevant]
    r = len(golden.relevant)
    at_r = retrieved[:r]
    r_precision = sum(doc in golden.relevant for doc in at_r) / r if r else 0.0
    recall = len(hits) / r if r else 0.0
    reciprocal = next(
        (1.0 / rank for rank, doc in enumerate(retrieved, 1) if doc in golden.relevant), 0.0
    )
    distractors = sum(doc in golden.distractors for doc in retrieved) / len(retrieved)
    return r_precision, recall, reciprocal, distractors


def evaluate(
    investigator: Investigator, goldens: GoldenSet, strategy: str, top_k: int = 3
) -> Metrics:
    """Run one retrieval strategy over every golden query and average the results."""
    totals = [0.0, 0.0, 0.0, 0.0]
    for golden in goldens.queries:
        retrieved = _retrieve(investigator, golden.question, strategy, top_k)
        for i, value in enumerate(score_query(retrieved, golden)):
            totals[i] += value
    n = max(len(goldens.queries), 1)
    precision, recall, mrr, distractor_rate = (total / n for total in totals)
    return Metrics(
        strategy=strategy,
        precision=precision,
        recall=recall,
        mrr=mrr,
        distractor_rate=distractor_rate,
    )


def _retrieve(investigator: Investigator, question: str, strategy: str, top_k: int) -> list[str]:
    pool = max(top_k * 4, 10)
    if strategy == "dense":
        embedding = investigator.embedder.embed([question])[0]
        return documents_in_order(investigator.vector_index.search(embedding, pool), top_k)
    if strategy == "lexical":
        return documents_in_order(investigator.lexical_index.search(question, pool), top_k)
    if strategy == "hybrid":
        return documents_in_order(investigator.hybrid_search(question, pool), top_k)
    if strategy == "hybrid+rerank":
        candidates = investigator.hybrid_search(question, pool)
        return documents_in_order(investigator.rerank(question, candidates), top_k)
    if strategy in ("pipeline:single", "pipeline:agentic"):
        # Not a ranking but a decision: which sources the finished report actually rests
        # on. The stages above are judged on ordering; the pipeline is judged on what it
        # let through, which is the number a reader of the report is exposed to.
        mode = cast("Mode", strategy.split(":", 1)[1])
        investigation = investigator.investigate(question, mode=mode)
        return investigation.sources
    raise ValueError(f"unknown strategy: {strategy}")


#: Retriever stages only — cheap, deterministic, and what CI benchmarks.
STRATEGIES = ("dense", "lexical", "hybrid", "hybrid+rerank")

#: End-to-end modes. These spend LLM calls per query, so they are opt-in.
PIPELINE_STRATEGIES = ("pipeline:single", "pipeline:agentic")


def evaluate_all(
    investigator: Investigator,
    goldens: GoldenSet,
    top_k: int = 3,
    *,
    include_pipeline: bool = False,
) -> list[Metrics]:
    strategies = STRATEGIES + (PIPELINE_STRATEGIES if include_pipeline else ())
    return [evaluate(investigator, goldens, strategy, top_k) for strategy in strategies]


def format_table(results: Sequence[Metrics]) -> str:
    return "\n".join([HEADER, *(m.as_row() for m in results)])
