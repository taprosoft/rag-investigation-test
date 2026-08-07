# Technical report

Design decisions, rationale, and measured results. Setup and usage: [README.md](README.md).

---

## 1. The data decides the design

| File | Content | Verdict |
|---|---|---|
| case_1 | Phishing → stolen wallet credentials | **Relevant** — initial access |
| case_2 | Solana wallet, funds split across addresses | **Relevant** — laundering |
| case_3 | Credential stuffing, 20 failed logins | **Relevant** — reconnaissance |
| case_4 | Keylogger malware — *"no direct link … to the stolen funds"* | **Distractor** |
| case_5 | Tornado Cash mixer, five intermediary wallets | **Relevant** — laundering |
| case_6 | Blocked SQL injection — *"no connection to the stolen … funds"* | **Distractor** |
| case_7 | Bitcoin ransom note after the attack | **Partial** — extortion, not theft |
| case_8 | Metadata breach two weeks earlier — *"no direct connection"* | **Distractor** |

Three distractors state their own irrelevance; all eight share the same vocabulary. Finding
relevant-looking text is trivial here. **Rejecting text that is topically perfect and
evidentially worthless is the problem.** So the system is built around a precision gate, not
a better ranker.

---

## 2. Retrieval strategy

Both modes from the brief are implemented and switchable (`--mode`, and a radio in the UI).
Default is agentic.

**Single-step** — one hybrid search → rerank → threshold → synthesise. One LLM call, ~1s.
Right choice when the question already names what it wants; also the baseline agentic has to
beat.

**Multi-step agentic** — a closed loop with a stopping condition:

```
gap ← "nothing collected yet"
for round in 1..max_rounds:
    query, why ← PLANNER(question, queries tried, evidence so far, gap)
    candidates ← hybrid_search(query) → rerank(against the question)
    shortlist  ← above threshold, unseen, ≤2 per source file
    verdict    ← GATEKEEPER(question, evidence so far, shortlist)
                   → admit[]/reject[] with reasons, sufficient?, what's missing
    evidence  += admitted
    stop if sufficient, or evidence full, or nothing admitted after round 1
    gap ← verdict.gap
```

Why a loop rather than one-shot query expansion:

- **The stopping condition is the point.** Fan-out always returns *k* passages whether they
  deserve to exist or not. The loop can conclude "enough" or "the archive doesn't answer
  this" — both are real answers.
- **The gatekeeper sees the shortlist plus what's already collected**, so it can reject a
  passage as redundant, not just irrelevant.
- **Each round is steered by the named gap**, so round 2 is a targeted follow-up rather than
  a synonym of round 1.
- **It shows its work** — every query, admission and rejection carries a reason, surfaced in
  the UI trace and `--trace`. On an evidential task, "why is this here?" matters as much as
  the answer.

Rounds are capped so a bad plan can't loop forever. Reranking always scores against the
**original question**, never the round's query: the query finds material, but relevance is
only ever relevance to what was asked.

### Retrieval core (shared)

Dense (`text-embedding-3-small`) generalises over paraphrase. BM25 catches literal tokens
embeddings blur — IPs, wallet names, `203.0.113.7`. RRF fuses them without calibrating a
cosine against a BM25 score. Cohere `rerank-v3.5` reads query and passage jointly, which a
bi-encoder structurally cannot.

Plus a **per-source cap** (2 passages per file) on the shortlist. Ranking alone favours
whichever document repeats the query's vocabulary most. Not theoretical: before the cap,
`case_2` took three of five slots on the laundering question and **the Tornado Cash passage
never surfaced**. With it, it does.

### Storage: in-memory, deliberately

BM25 and vectors are in-process — a NumPy matrix and ~50 lines of Okapi BM25. At 8 files /
24 paragraphs, brute-force cosine is *exact* and instant; a vector DB adds a dependency, a
network hop and ops weight for no measurable gain.

Sized to the data, not a design limit. Both sit behind narrow interfaces (`VectorIndex`,
`BM25Index`), so the upgrade is a one-class change: **vectors** → Qdrant / Chroma / pgvector
/ OpenSearch kNN; **full text** → OpenSearch / Elasticsearch / Postgres FTS for real BM25 at
scale with analysers and phrase queries. Fusion, reranking, the agent loop and citations are
storage-agnostic and wouldn't change.

Vectors cache to `.cache/`, keyed on corpus content **and embedder identity** (§5).

---

## 3. Citations

Following OpenAI's
[citation-formatting convention](https://developers.openai.com/api/docs/guides/citation-formatting).
Evidence goes to the synthesiser as numbered sentences:

```
[E1] source: case_5
L1: The stolen cryptocurrency was quickly moved through Tornado Cash…
L2: The exchange's security team noticed that within 30 minutes…
```

The model writes prose with inline markers using private-use delimiters
(`citeE1L2`) that can't collide with legitimate output. We
extract with offsets, strip them, and validate each against the evidence supplied.

Why this beats asking for a list of quotes (the first design):

1. **Verification is exact.** Quote-matching needs normalisation — models swap curly
   apostrophes, collapse newlines — and normalisation lets paraphrases through while
   flagging good citations. A line locator either resolves or it doesn't.
2. **It resolves to a sentence**, which is the granularity the UI highlights and a reader
   wants.
3. **Placement carries meaning** — the marker sits where the model put it, so you see *which
   clause* is sourced.
4. **Failure is visible.** An unknown label or an out-of-range line renders `⚠ unverified`
   rather than passing as grounded. Every report prints a grounding rate; live runs here hit
   **100% of claims resolving to a specific source line**.

**In the UI:** clicking a chip scrolls to the evidence card and highlights the exact
sentence. The mapping is computed server-side from resolved line numbers — no browser text
search — so it's deterministic and unit-tested.

---

## 4. Results

Six labelled questions over the supplied archive (`eval/goldens_case.json`). Reproduce:
`python -m detective eval --goldens eval/goldens_case.json --pipeline`.

| strategy | R-precision | recall@k | MRR | **distractor rate** |
|---|---|---|---|---|
| dense | 0.83 | 0.83 | 0.83 | 0.11 |
| lexical (BM25) | 0.42 | 0.42 | 0.58 | 0.33 |
| hybrid (RRF) | 0.75 | 0.83 | 0.75 | 0.11 |
| hybrid + rerank | 0.58 | 0.75 | 0.72 | 0.28 |
| pipeline:single | 0.42 | 0.50 | 0.50 | 0.18 |
| **pipeline:agentic** | 0.42 | 0.42 | 0.50 | **0.00** |

**Read the last column.** Every retrieval-only strategy puts a known distractor in front of
the detective 11–33% of the time. The agentic pipeline never does — across all six
questions, zero distractors reached a report.

**Read the first four honestly: dense alone out-ranks the hybrid.** Not my prediction. On
short single-topic paragraphs `text-embedding-3-small` is strong, while BM25 over 24 chunks
sharing most of their vocabulary is noisy, and equal-weight RRF let the weaker ranker drag
the stronger one down. Measuring it led to weighting fusion 2:1 toward dense, recovering
R-precision 0.58 → 0.75 — but dense alone is still ahead, and the table says so.

The reranker also doesn't improve document ranking here (0.58 vs 0.75). Its value is
different and real: it emits a *calibrated* 0–1 score, which is what makes an absolute
cut-off meaningful. Fused RRF scores can't — they're relative to the result set. The
threshold is the gate; the reranker is what makes a gate possible.

The pipeline trades recall for precision (0.42 vs 0.83): the gatekeeper is conservative and
`max_evidence` caps the set. For this task that's the right trade — three solid passages beat
a set that also cites a file announcing its own irrelevance — but it *is* a trade, and
`rerank_threshold` / `max_evidence` are the dials.

**Worked example** — `How did the hacker launder the stolen funds?` admits `case_5#2`,
`case_2#0`, `case_5#1` (Tornado Cash, Solana wallet, transaction splitting) and rejects:

- `case_2#2` at 0.69 — *"generic and speculative … says the hacker 'likely' used an automated
  script"* — the **highest-ranked passage of all**, rejected on reasoning
- `case_1#1` at 0.47 — *"concerns how the attacker gained access, not how funds were
  laundered"*
- `case_4#1`, `case_6#2`, `case_1#0` — below threshold

That first rejection is the argument in one line: the best-ranked passage wasn't the best
evidence, and only a stage that *reads* it could tell.

**CI benchmark.** `eval/synthetic/` is a fictional 10-document archive with the same
adversarial structure, scored with deterministic stand-ins so numbers move only when the
architecture changes — never because a model drifted. Published to the job summary each run.

---

## 5. Notes

**Case-agnostic prompts.** All prompts are written for a generic document archive; the
corpus directory is a runtime parameter. `tests/test_prompts.py` tokenises both corpora and
fails the build if their vocabulary appears in any prompt.

**Two bugs the live runs caught**, both invisible to the test suite:

- *Vector cache poisoning* — the cache keyed on the configured model *name*, which
  `--offline` leaves untouched while producing 256-dim stand-in vectors. A later live run
  loaded them and crashed on a 1536-dim query. Fixed by adding `identity` to the `Embedder`
  protocol, which kills the class of bug rather than the instance.
- *Markers leaking into prose* — `open_questions` skipped the citation parser, rendering raw
  delimiters on screen. Also fixed: entries the model both cited and listed as "excluded" are
  dropped, so the report can't contradict itself.

**Limitations**

- Six and five labelled questions — enough to catch regressions, too few for confidence
  intervals. Treat single-point differences as noise.
- `dense_weight = 2.0` is tuned on this corpus. Documented as measured, not principled;
  re-measure on new data.
- Recall is agentic mode's weak spot — it stopped after reporting phishing (`case_1`) on the
  access question without the credential-stuffing precursor (`case_3`). Defensible, but a
  recall-oriented deployment would raise `max_evidence` and lower the threshold.
- No live-API test in CI, by design. Provider-contract drift is only caught on a live run.
- Sentence splitting is a regex — fine here, would need a real segmenter for abbreviations
  and decimals.
