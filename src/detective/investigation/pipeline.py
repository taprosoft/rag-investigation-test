"""The investigation pipeline: index, retrieve, verify, synthesise.

Two retrieval modes share one retrieval core and one synthesiser.

``single`` — one hybrid search on the question, reranked and thresholded. One LLM call,
sub-second, and the right default when the question already names what it wants.

``agentic`` — the model plans a query, reads what came back, rules on each candidate, and
either stops or names the gap it still needs to close. It costs a few calls per round, and
buys two things single-shot retrieval cannot: recall across passages that answer the same
question in different vocabulary, and an explicit admit/reject decision on each passage
that survives the reranker. On an archive seeded with plausible-but-irrelevant material,
that second point is the difference between a report and a plausible-sounding mistake.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from detective.core.config import Settings
from detective.core.corpus import chunk_documents, corpus_fingerprint, load_documents
from detective.core.models import (
    Chunk,
    Document,
    Evidence,
    Finding,
    Investigation,
    Report,
    ScoredChunk,
)
from detective.investigation.citations import format_evidence, parse_claim
from detective.investigation.prompts import (
    ASSESSOR_SYSTEM,
    ASSESSOR_USER,
    PLANNER_SYSTEM,
    PLANNER_USER,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER,
)
from detective.providers.base import ChatModel, Embedder, Reranker, parse_json_object
from detective.retrieval import BM25Index, VectorIndex, reciprocal_rank_fusion

Mode = Literal["single", "agentic"]

NO_EVIDENCE_SUMMARY = "No passage in the archive met the relevance bar for this question."


@dataclass(frozen=True, slots=True)
class Providers:
    """The three model dependencies, bundled so callers wire them once."""

    embedder: Embedder
    reranker: Reranker
    chat: ChatModel


class Investigator:
    """Owns the indexed corpus and answers questions against it."""

    def __init__(
        self,
        documents: Sequence[Document],
        chunks: Sequence[Chunk],
        vectors: VectorIndex,
        lexical: BM25Index,
        providers: Providers,
        settings: Settings,
    ) -> None:
        self.documents = list(documents)
        self.chunks = list(chunks)
        self._vectors = vectors
        self._lexical = lexical
        self._providers = providers
        self._settings = settings

    @property
    def vector_index(self) -> VectorIndex:
        return self._vectors

    @property
    def lexical_index(self) -> BM25Index:
        return self._lexical

    @property
    def embedder(self) -> Embedder:
        return self._providers.embedder

    # -- construction ------------------------------------------------------------------

    @classmethod
    def build(
        cls, providers: Providers, settings: Settings, *, use_cache: bool = True
    ) -> Investigator:
        """Load the corpus and build both indexes, reusing cached vectors when valid."""
        documents = load_documents(settings.corpus_dir)
        chunks = chunk_documents(documents)
        cache_path = _cache_path(settings, chunks, providers.embedder.identity)

        vectors = VectorIndex.load(cache_path, chunks) if use_cache else None
        if vectors is None:
            embeddings = providers.embedder.embed([c.text for c in chunks])
            vectors = VectorIndex(chunks, embeddings)
            if use_cache:
                vectors.save(cache_path)

        return cls(documents, chunks, vectors, BM25Index(chunks), providers, settings)

    # -- retrieval ---------------------------------------------------------------------

    def hybrid_search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Dense and lexical search fused by reciprocal rank."""
        dense = self._vectors.search(self._providers.embedder.embed([query])[0], top_k)
        lexical = self._lexical.search(query, top_k)
        return reciprocal_rank_fusion(
            [dense, lexical], top_k=top_k, weights=[self._settings.dense_weight, 1.0]
        )

    def rerank(self, question: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Score candidates against the question with the cross-encoder.

        Always against the *question*, never the round's query: the query is a means of
        finding material, but relevance is only ever relevance to what was asked.
        """
        if not candidates:
            return []
        ranked = self._providers.reranker.rerank(
            question, [c.chunk.text for c in candidates], top_n=len(candidates)
        )
        return [
            ScoredChunk(chunk=candidates[i].chunk, score=score, scorer="rerank")
            for i, score in ranked
        ]

    # -- investigation -----------------------------------------------------------------

    def investigate(self, question: str, mode: Mode = "agentic") -> Investigation:
        investigation = Investigation(question=question, mode=mode)
        if mode == "single":
            self._single_step(investigation)
        else:
            self._agentic(investigation)
        investigation.report = self._synthesise(investigation)
        return investigation

    def _single_step(self, investigation: Investigation) -> None:
        from detective.core.models import RoundResult

        question = investigation.question
        reranked = self.rerank(question, self.hybrid_search(question, self._settings.candidates))
        above = [s for s in reranked if s.score >= self._settings.rerank_threshold]
        keep = {s.chunk.chunk_id for s in self._shortlist(above)[: self._settings.max_evidence]}

        admitted: list[Evidence] = []
        rejected: list[tuple[str, float, str]] = []
        for scored in reranked:
            if scored.chunk.chunk_id not in keep:
                rejected.append(
                    (
                        scored.chunk.chunk_id,
                        scored.score,
                        f"relevance {scored.score:.2f} below threshold "
                        f"{self._settings.rerank_threshold:.2f}"
                        if scored.score < self._settings.rerank_threshold
                        else "outranked within the evidence limit",
                    )
                )
                continue
            admitted.append(
                Evidence(
                    label=f"E{len(admitted) + 1}",
                    chunk=scored.chunk,
                    rerank_score=scored.score,
                    round_index=1,
                    query=question,
                    rationale=f"top-ranked for the question (relevance {scored.score:.2f})",
                )
            )

        investigation.evidence = admitted
        investigation.rounds = [
            RoundResult(
                index=1,
                query=question,
                reason="single-step retrieval on the question as asked",
                candidates=tuple(reranked),
                admitted=tuple(admitted),
                rejected=tuple(rejected),
                sufficient=True,
                gap="",
            )
        ]

    def _shortlist(
        self, ranked: Sequence[ScoredChunk], already_used: Sequence[Chunk] = ()
    ) -> list[ScoredChunk]:
        """Take the best passages, but no more than ``per_document_limit`` from one file.

        Relevance ranking alone is quietly biased towards whichever document repeats the
        query's vocabulary most often, and on a small archive that means one file can fill
        the shortlist while a second file holding the other half of the answer never
        reaches the gatekeeper at all.
        """
        used: dict[str, int] = {}
        for chunk in already_used:
            used[chunk.doc_id] = used.get(chunk.doc_id, 0) + 1

        picked: list[ScoredChunk] = []
        for scored in ranked:
            if len(picked) >= self._settings.per_round_evidence:
                break
            doc_id = scored.chunk.doc_id
            if used.get(doc_id, 0) >= self._settings.per_document_limit:
                continue
            used[doc_id] = used.get(doc_id, 0) + 1
            picked.append(scored)
        return picked

    def _agentic(self, investigation: Investigation) -> None:
        from detective.core.models import RoundResult

        question = investigation.question
        tried: list[str] = []
        gap = "nothing collected yet"

        for round_index in range(1, self._settings.max_rounds + 1):
            query, reason = self._plan(question, tried, investigation.evidence, gap)
            tried.append(query)

            reranked = self.rerank(question, self.hybrid_search(query, self._settings.candidates))
            seen = {e.chunk.chunk_id for e in investigation.evidence}
            fresh = self._shortlist(
                [
                    s
                    for s in reranked
                    if s.score >= self._settings.rerank_threshold and s.chunk.chunk_id not in seen
                ],
                already_used=[e.chunk for e in investigation.evidence],
            )

            verdict = self._assess(question, investigation.evidence, fresh)
            admitted: list[Evidence] = []
            for scored in fresh:
                rationale = verdict.admitted.get(scored.chunk.chunk_id)
                if rationale is None:
                    continue
                admitted.append(
                    Evidence(
                        label=f"E{len(investigation.evidence) + len(admitted) + 1}",
                        chunk=scored.chunk,
                        rerank_score=scored.score,
                        round_index=round_index,
                        query=query,
                        rationale=rationale,
                    )
                )

            below_threshold = [
                (s.chunk.chunk_id, s.score, f"relevance {s.score:.2f} below threshold")
                for s in reranked
                if s.score < self._settings.rerank_threshold
            ][:3]
            ruled_out = [
                (s.chunk.chunk_id, s.score, verdict.rejected.get(s.chunk.chunk_id, "not admitted"))
                for s in fresh
                if s.chunk.chunk_id not in verdict.admitted
            ]

            investigation.evidence.extend(admitted[: self._settings.max_evidence])
            full = len(investigation.evidence) >= self._settings.max_evidence
            # A round that admits nothing new is the signal that the archive is exhausted;
            # allowing one such round protects against a weak opening query.
            sufficient = verdict.sufficient or full or (not admitted and round_index > 1)
            gap = verdict.gap or ""

            investigation.rounds.append(
                RoundResult(
                    index=round_index,
                    query=query,
                    reason=reason,
                    candidates=tuple(reranked),
                    admitted=tuple(admitted),
                    rejected=tuple(ruled_out + below_threshold),
                    sufficient=sufficient,
                    gap=gap,
                )
            )
            if sufficient:
                break

    # -- LLM steps ---------------------------------------------------------------------

    def _plan(
        self, question: str, tried: Sequence[str], collected: Sequence[Evidence], gap: str
    ) -> tuple[str, str]:
        """Ask the planner for the next query.

        The opening round always searches the question verbatim — spending a model call to
        rephrase a question nobody has tried yet is pure latency.
        """
        if not tried:
            return question, "opening search using the question as asked"
        raw = self._providers.chat.complete(
            PLANNER_SYSTEM,
            PLANNER_USER.format(
                question=question,
                tried="\n".join(f"- {t}" for t in tried) or "(none)",
                collected=format_evidence(collected, numbered=False) or "(none)",
                gap=gap or "(unspecified)",
            ),
            as_json=True,
        )
        parsed = parse_json_object(raw)
        query = str(parsed.get("query") or "").strip()
        reason = str(parsed.get("reason") or "").strip()
        if not query or query in tried:
            return (
                f"{question} {gap}".strip(),
                "planner returned nothing new; widened the search instead",
            )
        return query, reason or "planner did not give a reason"

    def _assess(
        self, question: str, collected: Sequence[Evidence], candidates: Sequence[ScoredChunk]
    ) -> _Verdict:
        """Have the model rule on each candidate and judge sufficiency."""
        if not candidates:
            return _Verdict({}, {}, sufficient=bool(collected), gap="no new passages found")
        raw = self._providers.chat.complete(
            ASSESSOR_SYSTEM,
            ASSESSOR_USER.format(
                question=question,
                collected=format_evidence(collected, numbered=False) or "(none)",
                candidates="\n\n".join(
                    f"[{c.chunk.chunk_id}] (relevance {c.score:.2f})\n{c.chunk.text}"
                    for c in candidates
                ),
            ),
            as_json=True,
        )
        parsed = parse_json_object(raw)
        valid = {c.chunk.chunk_id for c in candidates}
        return _Verdict(
            admitted=_rulings(parsed.get("admit"), valid),
            rejected=_rulings(parsed.get("reject"), valid),
            sufficient=bool(parsed.get("sufficient", False)),
            gap=str(parsed.get("gap") or ""),
        )

    def _synthesise(self, investigation: Investigation) -> Report:
        """Generate the report and validate every citation against the passages supplied."""
        if not investigation.evidence:
            return Report(
                summary=NO_EVIDENCE_SUMMARY,
                findings=(),
                timeline=(),
                open_questions=("Does the archive hold material on this question at all?",),
            )
        raw = self._providers.chat.complete(
            SYNTHESIS_SYSTEM,
            SYNTHESIS_USER.format(
                question=investigation.question,
                evidence=format_evidence(investigation.evidence),
            ),
            as_json=True,
        )
        return build_report(parse_json_object(raw), investigation.evidence)


@dataclass(frozen=True, slots=True)
class _Verdict:
    admitted: dict[str, str]
    rejected: dict[str, str]
    sufficient: bool
    gap: str


# --------------------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------------------


def build_report(parsed: dict[str, object], evidence: Sequence[Evidence]) -> Report:
    """Turn the synthesiser's JSON into a ``Report`` with validated citations."""
    summary_text, _ = parse_claim(str(parsed.get("summary") or ""), evidence)

    def claims(key: str) -> tuple[Finding, ...]:
        findings: list[Finding] = []
        for raw in _strings(parsed.get(key)):
            statement, citations = parse_claim(raw, evidence)
            if statement:
                findings.append(Finding(statement=statement, citations=citations))
        return tuple(findings)

    findings, timeline = claims("findings"), claims("timeline")
    # Models sometimes drop a marker into a free-text field the schema never asked to
    # cite. Strip rather than render: a raw delimiter on screen is a bug either way.
    raw_questions = _strings(parsed.get("open_questions"))
    open_questions = tuple(
        text for text in (parse_claim(q, evidence)[0] for q in raw_questions) if text
    )

    cited = {c.label for claim in timeline + findings for c in claim.citations}
    excluded = tuple(
        (label, reason)
        for label, reason in (
            (
                str(item.get("label") or item.get("id") or ""),
                parse_claim(str(item.get("reason") or ""), evidence)[0],
            )
            for item in _items(parsed.get("excluded"))
        )
        # A passage the report leans on is not excluded, whatever the model also claimed;
        # showing both would put a visible contradiction in front of the reader.
        if (label or reason) and label not in cited
    )
    return Report(
        summary=summary_text,
        findings=findings,
        timeline=timeline,
        open_questions=open_questions,
        excluded=excluded,
    )


def _items(value: object) -> list[dict[str, object]]:
    return [item for item in _sequence(value) if isinstance(item, dict)]


def _strings(value: object) -> list[str]:
    return [item for item in _sequence(value) if isinstance(item, str)]


def _sequence(value: object) -> list[object]:
    return cast("list[object]", value) if isinstance(value, list) else []


def _rulings(value: object, valid_ids: set[str]) -> dict[str, str]:
    """Keep only rulings that name a candidate we actually offered."""
    rulings: dict[str, str] = {}
    for item in _items(value):
        chunk_id = str(item.get("id") or "").strip()
        if chunk_id in valid_ids:
            rulings[chunk_id] = str(item.get("reason") or "").strip() or "no reason given"
    return rulings


def _cache_path(settings: Settings, chunks: Sequence[Chunk], embedder_id: str) -> Path:
    """Cache file keyed on corpus content *and* the embedder that produced the vectors."""
    fingerprint = corpus_fingerprint(list(chunks), embedder_id)
    return settings.cache_dir / f"index-{fingerprint}.npz"
